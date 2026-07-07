import time
import random
import json
import sys
import argparse
import paho.mqtt.client as mqtt

# ============================================================
# WEEK 3 PRIVATE SHIELD - FAKE IOT DEVICE SIMULATOR
# This file creates fake IoT devices and sends data to MQTT.
# Week 3 upgrade: it now sends network-flow features for ML.
# ============================================================


# ============================================================
# 1. DEVICE PROFILES
# Each device has different normal behavior.
# Camera = high traffic
# Sensor = low traffic
# Plug = occasional traffic
# ============================================================

DEVICE_PROFILES = {
    "camera": {
        "device_id": "cam_01",
        "port": 554,                 # Camera usually uses streaming-related ports
        "packets_per_sec": 10,       # Camera sends many packets
        "bytes_per_pkt": 1400,       # Camera packets are large
        "sleep_time": 0.1            # 0.1 sec delay = around 10 messages per second
    },
    "sensor": {
        "device_id": "sensor_02",
        "port": 1883,                # Sensor uses MQTT port
        "packets_per_sec": 1,        # Sensor sends slow small messages
        "bytes_per_pkt": 32,
        "sleep_time": 1              # 1 message per second
    },
    "plug": {
        "device_id": "plug_03",
        "port": 80,                  # Smart plug example port
        "packets_per_sec": 0.2,      # Very low traffic
        "bytes_per_pkt": 256,
        "sleep_time": 5              # 1 message every 5 seconds
    }
}


# ============================================================
# 2. LABEL MAP
# These labels help Person B create labelled.csv for ML.
# 0 = normal traffic
# 1 = DoS attack traffic
# 2 = port scan traffic
# 3 = Mirai-style traffic
# ============================================================

LABELS = {
    "normal": 0,
    "dos": 1,
    "port_scan": 2,
    "mirai": 3
}


# ============================================================
# 3. NORMAL FEATURE GENERATOR
# This creates realistic low-value network features.
# These features are similar to N-BaIoT/network-flow columns.
# ============================================================

def generate_normal_features(profile):
    flow_duration = round(random.uniform(1.0, 10.0), 2)
    fwd_packets = random.randint(1, 20)
    bwd_packets = random.randint(1, 10)

    return {
        "flow_duration": flow_duration,                             # How long the network flow lasts
        "fwd_packets": fwd_packets,                                 # Packets sent forward
        "bwd_packets": bwd_packets,                                 # Packets received back
        "flow_bytes_per_sec": round(profile["bytes_per_pkt"] * profile["packets_per_sec"], 2),
        "flow_pkts_per_sec": profile["packets_per_sec"],
        "fwd_pkt_len_mean": profile["bytes_per_pkt"],               # Average forward packet size
        "bwd_pkt_len_mean": random.randint(20, 200),                # Average backward packet size
        "fin_flag_cnt": random.randint(0, 1),                       # TCP FIN flag count
        "syn_flag_cnt": random.randint(0, 2),                       # TCP SYN flag count
        "rst_flag_cnt": random.randint(0, 1)                        # TCP RST flag count
    }


# ============================================================
# 4. ATTACK FEATURE GENERATOR
# This creates extreme traffic values.
# ML models can learn these are different from normal behavior.
# ============================================================

def generate_attack_features(mode):
    # DoS attack = too many packets and too much traffic
    if mode == "dos":
        return {
            "flow_duration": round(random.uniform(0.1, 2.0), 2),
            "fwd_packets": random.randint(500, 2000),
            "bwd_packets": random.randint(0, 5),
            "flow_bytes_per_sec": random.randint(500000, 2000000),
            "flow_pkts_per_sec": random.randint(500, 2000),
            "fwd_pkt_len_mean": random.randint(800, 1500),
            "bwd_pkt_len_mean": random.randint(0, 50),
            "fin_flag_cnt": 0,
            "syn_flag_cnt": random.randint(100, 500),
            "rst_flag_cnt": random.randint(20, 100)
        }

    # Port scan = many connection attempts, many SYN/RST flags
    if mode == "port_scan":
        return {
            "flow_duration": round(random.uniform(0.1, 1.0), 2),
            "fwd_packets": random.randint(50, 300),
            "bwd_packets": random.randint(0, 20),
            "flow_bytes_per_sec": random.randint(10000, 100000),
            "flow_pkts_per_sec": random.randint(100, 500),
            "fwd_pkt_len_mean": random.randint(40, 100),
            "bwd_pkt_len_mean": random.randint(0, 80),
            "fin_flag_cnt": random.randint(10, 100),
            "syn_flag_cnt": random.randint(50, 300),
            "rst_flag_cnt": random.randint(50, 300)
        }

    # Mirai-style traffic = botnet-like repeated connections
    if mode == "mirai":
        return {
            "flow_duration": round(random.uniform(0.5, 5.0), 2),
            "fwd_packets": random.randint(200, 1000),
            "bwd_packets": random.randint(0, 30),
            "flow_bytes_per_sec": random.randint(100000, 800000),
            "flow_pkts_per_sec": random.randint(200, 1000),
            "fwd_pkt_len_mean": random.randint(300, 1200),
            "bwd_pkt_len_mean": random.randint(20, 150),
            "fin_flag_cnt": random.randint(0, 5),
            "syn_flag_cnt": random.randint(100, 400),
            "rst_flag_cnt": random.randint(10, 80)
        }


# ============================================================
# 5. COMMAND-LINE ARGUMENTS
# This lets you choose device type and traffic mode from terminal.
#
# Example:
# python fake_device.py --type camera --mode normal
# python fake_device.py --type camera --mode dos
# ============================================================

parser = argparse.ArgumentParser(description="PRIVATE SHIELD Week 3 Fake IoT Device Simulator")

parser.add_argument(
    "--type",
    required=True,
    choices=["camera", "sensor", "plug"],
    help="Choose device type: camera, sensor, or plug"
)

parser.add_argument(
    "--mode",
    default="normal",
    choices=["normal", "dos", "port_scan", "mirai"],
    help="Choose traffic mode: normal, dos, port_scan, or mirai"
)

args = parser.parse_args()

device_type = args.type
traffic_mode = args.mode
profile = DEVICE_PROFILES[device_type]
device_id = profile["device_id"]


# ============================================================
# 6. MQTT SETTINGS
# MQTT broker is Mosquitto running on your own laptop.
# The topic will look like:
# devices/cam_01
# devices/sensor_02
# devices/plug_03
# ============================================================

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = f"devices/{device_id}"


# ============================================================
# 7. CREATE MQTT CLIENT
# This version supports both newer and older paho-mqtt versions.
# ============================================================

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()


# ============================================================
# 8. CONNECT TO MQTT BROKER
# If Mosquitto is not running, this part will show an error.
# ============================================================

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    print(f"[{device_id}] Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    print(f"[{device_id}] Device type: {device_type}")
    print(f"[{device_id}] Traffic mode: {traffic_mode}")
    print(f"[{device_id}] Label: {LABELS[traffic_mode]}")
    print(f"[{device_id}] Publishing to topic: {MQTT_TOPIC}")

except Exception as e:
    print(f"[{device_id}] Failed to connect: {e}")
    sys.exit(1)


# ============================================================
# 9. MAIN LOOP
# This runs forever until you press Ctrl+C.
# Every loop:
# 1. Generate normal or attack features
# 2. Build JSON payload
# 3. Publish to MQTT
# 4. Print the message
# 5. Sleep based on device profile
# ============================================================

try:
    while True:
        if traffic_mode == "normal":
            flow_features = generate_normal_features(profile)
        else:
            flow_features = generate_attack_features(traffic_mode)

        payload = {
            "device_id": device_id,
            "device_type": device_type,
            "mode": traffic_mode,
            "label": LABELS[traffic_mode],

            # Basic IoT telemetry
            "temp": round(random.uniform(20.0, 35.0), 2),
            "humidity": round(random.uniform(40.0, 80.0), 2),
            "packets_per_sec": profile["packets_per_sec"],
            "bytes_per_pkt": profile["bytes_per_pkt"],
            "port": profile["port"],

            # Week 3 ML/network-flow features
            **flow_features
        }

        json_payload = json.dumps(payload)

        client.publish(MQTT_TOPIC, json_payload)

        print(f"[{device_id}] Published: {json_payload}")

        time.sleep(profile["sleep_time"])

except KeyboardInterrupt:
    print(f"\n[{device_id}] Device stopped by user.")

finally:
    client.loop_stop()
    client.disconnect()
    print(f"[{device_id}] Disconnected from MQTT broker.")