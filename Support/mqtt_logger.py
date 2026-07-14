import sys
import os
import time
import json
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt


BROKER_HOST = "localhost"
BROKER_PORT = 1883

DEVICE_TOPIC = "devices/#"
CONTROL_TOPIC = "private_shield/attack_mode"

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "labelled.csv")
STATUS_PATH = os.path.join(DATA_DIR, "status.json")

LABEL_MAP = {
    "normal": 0,
    "dos": 1,
    "port_scan": 2,
    "mirai": 3,
}

CSV_COLUMNS = [
    "timestamp",
    "topic",
    "device_id",
    "device_type",
    "temperature",
    "temp",
    "humidity",
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
    "rst_flag_cnt",
    "sequence",
    "generated_at",
    "attack_type",
    "label",
    "raw_payload",
]

BATCH_SIZE = 100

os.makedirs(DATA_DIR, exist_ok=True)

total_messages_count = 0
message_buffer = []
message_history = []
last_message_data = "N/A"

current_mode = "normal"
current_label = 0


def normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "portscan":
        return "port_scan"
    return mode


def flush_buffer_to_csv():
    global message_buffer

    if not message_buffer:
        return

    df = pd.DataFrame(message_buffer)
    df = df.reindex(columns=CSV_COLUMNS)

    file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)

    message_buffer.clear()


def update_status_json(running=True):
    global message_history

    current_time = time.time()
    message_history = [
        item for item in message_history
        if item["time"] > current_time - 10
    ]

    devices_online = sorted(
        set(item["device_id"] for item in message_history)
    )

    status_data = {
        "running": running,
        "topic": DEVICE_TOPIC,
        "control_topic": CONTROL_TOPIC,
        "normal_csv": CSV_PATH,
        "labelled_csv": CSV_PATH,
        "current_mode": current_mode,
        "current_label": current_label,
        "total_messages": total_messages_count,
        "messages_last_10s": len(message_history),
        "devices_online": devices_online,
        "alerts_fired": 0,
        "last_msg": last_message_data,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as file:
            json.dump(status_data, file, indent=4)
    except Exception as error:
        print(f"Error updating status.json: {error}")


def on_connect(client, userdata, flags, rc, properties=None):
    connection_ok = (
        rc == 0
        or getattr(rc, "value", None) == 0
    )

    if connection_ok:
        print(
            f"Connected successfully to broker at "
            f"{BROKER_HOST}:{BROKER_PORT}"
        )

        client.subscribe(DEVICE_TOPIC)
        client.subscribe(CONTROL_TOPIC)

        print(f"Subscribed to device topic: {DEVICE_TOPIC}")
        print(f"Subscribed to control topic: {CONTROL_TOPIC}")
        print(
            f"Current mode: {current_mode} | "
            f"Current label: {current_label}"
        )
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    global total_messages_count
    global message_buffer
    global message_history
    global last_message_data
    global current_mode
    global current_label

    try:
        payload_str = msg.payload.decode("utf-8")
        payload_dict = json.loads(payload_str)
    except Exception as error:
        print(
            f"Skipping invalid JSON message on topic "
            f"{msg.topic}: {error}"
        )
        return

    if msg.topic == CONTROL_TOPIC:
        received_mode = normalize_mode(
            payload_dict.get("mode", "normal")
        )

        if received_mode not in LABEL_MAP:
            print(
                f"Ignoring unknown attack mode: "
                f"{received_mode}"
            )
            return

        current_mode = received_mode
        current_label = LABEL_MAP[received_mode]

        print("\n" + "=" * 55)
        print(
            f"MODE CHANGED: {current_mode.upper()} "
            f"| LABEL = {current_label}"
        )
        print("=" * 55)

        update_status_json(running=True)
        return

    if not msg.topic.startswith("devices/"):
        return

    current_time = time.time()

    device_id = payload_dict.get("device_id")
    if not device_id:
        topic_parts = msg.topic.split("/")
        device_id = (
            topic_parts[-1]
            if len(topic_parts) > 1
            else "unknown"
        )

    message_history.append({
        "time": current_time,
        "device_id": device_id,
    })

    last_message_data = payload_dict

    payload_attack_type = normalize_mode(
        payload_dict.get("attack_type", "")
    )

    if (
        payload_attack_type in LABEL_MAP
        and payload_attack_type != "normal"
    ):
        row_attack_type = payload_attack_type
        row_label = LABEL_MAP[payload_attack_type]
    else:
        row_attack_type = current_mode
        row_label = current_label

    row_data = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3],
        "topic": msg.topic,
        **payload_dict,
        "device_id": device_id,
        "attack_type": row_attack_type,
        "label": row_label,
        "raw_payload": json.dumps(
            payload_dict,
            ensure_ascii=False,
        ),
    }

    message_buffer.append(row_data)
    total_messages_count += 1

    if total_messages_count <= 5 or total_messages_count % 100 == 0:
        print(
            f"Row {total_messages_count}: "
            f"device={device_id} | "
            f"mode={row_attack_type} | "
            f"label={row_label}"
        )

    if len(message_buffer) >= BATCH_SIZE:
        flush_buffer_to_csv()


def create_mqtt_client():
    try:
        return mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2
        )
    except AttributeError:
        return mqtt.Client()


if __name__ == "__main__":
    client = create_mqtt_client()

    client.on_connect = on_connect
    client.on_message = on_message

    print(
        f"Connecting to MQTT Broker at "
        f"{BROKER_HOST}:{BROKER_PORT}..."
    )

    try:
        client.connect(
            BROKER_HOST,
            BROKER_PORT,
            60,
        )
    except Exception as error:
        print("\n" + "=" * 50)
        print("ERROR: Could not connect to the MQTT Broker!")
        print(f"Details: {error}")
        print(
            "Please ensure that Mosquitto is running "
            "on localhost:1883."
        )
        print("=" * 50 + "\n")
        sys.exit(1)

    client.loop_start()
    update_status_json(running=True)

    print(
        "MQTT Logger is running. "
        "Collecting auto-labelled data."
    )
    print(f"Output file: {CSV_PATH}")
    print("Press Ctrl+C only after the full sequence finishes.")

    try:
        while True:
            time.sleep(1)
            update_status_json(running=True)

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Safely shutting down...")

    finally:
        print("Saving remaining buffered rows to CSV...")
        flush_buffer_to_csv()

        client.loop_stop()
        client.disconnect()

        update_status_json(running=False)

        if os.path.exists(CSV_PATH):
            try:
                df = pd.read_csv(CSV_PATH)

                print("\n" + "=" * 55)
                print(f"Collected {len(df)} rows in total.")

                print("\nLabel counts:")
                print(
                    df["label"]
                    .value_counts()
                    .sort_index()
                )

                print("\nAttack type counts:")
                print(
                    df["attack_type"]
                    .value_counts()
                )

                print("\nLast five rows:")
                print(
                    df[
                        [
                            "timestamp",
                            "device_id",
                            "attack_type",
                            "label",
                        ]
                    ].tail(5)
                )
                print("=" * 55)

            except Exception as error:
                print(
                    f"Error generating final dataset "
                    f"report: {error}"
                )
        else:
            print("No labelled CSV data file was created.")
