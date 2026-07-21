import json
import time
import os
from datetime import datetime

import pandas as pd
import joblib
import requests
import paho.mqtt.client as mqtt

# MQTT CONFIGURATION
# This is the local Mosquitto broker.
# fake_device.py publishes here.
# detector.py listens here.

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "devices/#"


# MODEL PATH
# should train the Random Forest model and save it here.
MODEL_PATH = "models/rf.pkl"


# OPTIONAL BACKEND API
#  may run FastAPI backend on this URL.
# If backend is running, detector.py can send alerts there.
# If backend is not running, detector.py will still work.

BACKEND_ALERT_URL = "http://127.0.0.1:8000/alert"


# FEATURE COLUMNS
# IMPORTANT:
# These columns must match the training columns used by Person B.
# The order must also be the same.

FEATURE_COLUMNS = [
    "packets_per_sec",
    "bytes_per_pkt",
    "port",
    "flow_duration",
    "fwd_packets",
    "bwd_packets",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt"
]
# LABEL MAP
# Model output number will be converted into readable attack name.
LABEL_TO_ATTACK = {
    0: "Normal",
    1: "DoS",
    2: "Port Scan",
    3: "Mirai"
}


# GLOBAL VARIABLES
# model = trained Random Forest model
# use_fallback = True means model file was not found

model = None
use_fallback = False


# LOAD MODEL FUNCTION
# This loads models/rf.pkl.
# If the file is missing, it uses a simple temporary rule-based detector.
# Final project should use the real ML model from Person B.

def load_model():
    global model, use_fallback

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        use_fallback = False
        print(f"[MODEL] Loaded trained model from {MODEL_PATH}")
    else:
        model = None
        use_fallback = True
        print(f"[WARNING] Model file not found: {MODEL_PATH}")
        print("[WARNING] Running temporary fallback detection mode.")
        print("[WARNING] Final version should use Person B's trained rf.pkl model.")


# EXTRACT FEATURES FUNCTION
# This converts raw MQTT JSON message into ML feature vector.
# Example:
# message JSON → pandas DataFrame → model prediction

def extract_features(message_dict):
    feature_row = {}

    for col in FEATURE_COLUMNS:
        # If any feature is missing, use 0 as default.
        # This prevents the detector from crashing.
        feature_row[col] = message_dict.get(col, 0)

    # Convert into DataFrame because sklearn models work well with this format.
    features_df = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)

    return features_df


# TEMPORARY FALLBACK DETECTOR
# This is only for testing when models/rf.pkl is not ready.
# It uses simple rules based on Week 3 generated feature ranges.

def fallback_predict(features_df):
    row = features_df.iloc[0]

    flow_pkts_per_sec = row["flow_pkts_per_sec"]
    flow_bytes_per_sec = row["flow_bytes_per_sec"]
    syn_flag_cnt = row["syn_flag_cnt"]
    rst_flag_cnt = row["rst_flag_cnt"]
    fwd_pkt_len_mean = row["fwd_pkt_len_mean"]

    # DoS usually has very high packet rate and very high byte rate.
    if flow_pkts_per_sec >= 500 and flow_bytes_per_sec >= 500000:
        return 1

    # Port scan usually has many SYN and RST flags with small packets.
    if syn_flag_cnt >= 50 and rst_flag_cnt >= 50 and fwd_pkt_len_mean <= 150:
        return 2

    # Mirai-style traffic usually has many repeated connections.
    if flow_pkts_per_sec >= 200 and syn_flag_cnt >= 100:
        return 3

    # Otherwise normal.
    return 0


# SEND ALERT TO BACKEND
# This helps Person D.
# If FastAPI backend is running, alert will be sent to dashboard.
# If not running, detector.py continues normally.

def send_alert_to_backend(alert_data):
    try:
        requests.post(BACKEND_ALERT_URL, json=alert_data, timeout=1)
    except Exception:
        # Backend may not be running yet.
        # We silently ignore this so detector does not stop.
        pass


# SAVE ALERT LOCALLY
# This saves alerts to logs/alerts.jsonl.
# JSONL means one JSON object per line.

def save_alert(alert_data):
    os.makedirs("logs", exist_ok=True)

    with open("logs/alerts.jsonl", "a") as f:
        f.write(json.dumps(alert_data) + "\n")


# MQTT CONNECT CALLBACK
# This runs when detector connects to the MQTT broker.

def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] Connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
    print(f"[MQTT] Subscribing to topic: {MQTT_TOPIC}")

    client.subscribe(MQTT_TOPIC)


# MQTT MESSAGE CALLBACK
# This runs every time a fake device publishes a message.
# Main detection happens here.

def on_message(client, userdata, msg):
    try:
        # Decode MQTT payload from bytes to string.
        payload_text = msg.payload.decode("utf-8")

        # Convert JSON string to Python dictionary.
        message_dict = json.loads(payload_text)

        # Extract useful identity fields.
        device_id = message_dict.get("device_id", "unknown_device")
        device_type = message_dict.get("device_type", "unknown_type")

        # Convert raw JSON into ML feature vector.
        features_df = extract_features(message_dict)

        # Predict using trained model or temporary fallback.
        if use_fallback:
            prediction = fallback_predict(features_df)
        else:
            prediction = int(model.predict(features_df)[0])

        attack_name = LABEL_TO_ATTACK.get(prediction, "Unknown")

        # Print normal traffic shortly.
        if prediction == 0:
            print(f"[NORMAL] {device_id} ({device_type}) traffic is normal.")

        # Print alert for attack.
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            alert_data = {
                "timestamp": timestamp,
                "device_id": device_id,
                "device_type": device_type,
                "attack_type": attack_name,
                "prediction": prediction,
                "topic": msg.topic,
                "flow_pkts_per_sec": float(features_df.iloc[0]["flow_pkts_per_sec"]),
                "flow_bytes_per_sec": float(features_df.iloc[0]["flow_bytes_per_sec"]),
                "syn_flag_cnt": float(features_df.iloc[0]["syn_flag_cnt"]),
                "rst_flag_cnt": float(features_df.iloc[0]["rst_flag_cnt"])
            }

            print("\n==================== ALERT ====================")
            print(f"Time        : {timestamp}")
            print(f"Device      : {device_id}")
            print(f"Device Type : {device_type}")
            print(f"Attack Type : {attack_name}")
            print(f"Topic       : {msg.topic}")

            save_alert(alert_data)
            send_alert_to_backend(alert_data)

    except Exception as e:
        print(f"[ERROR] Failed to process message: {e}")


# MAIN FUNCTION
# This starts the detector.

def main():
    print("PRIVATE SHIELD - Week 4 Real-Time Detector")

    load_model()

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"[ERROR] Could not connect to MQTT broker: {e}")
        print("[FIX] Start Mosquitto first using: mosquitto -v")
        return

    print("[SYSTEM] Detector is running. Press Ctrl+C to stop.")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Detector stopped by user.")


if __name__ == "__main__":
    main()