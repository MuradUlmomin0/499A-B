import os
import time
import json

STATUS_FILE = "data/status.json"

def clear_screen():
    print("\033c", end="")

def load_status():
    default_status = {
        "running": False,
        "total_messages": 0,
        "messages_last_10s": 0,
        "devices_online": [],
        "alerts_fired": 0,
        "last_msg": {},
        "updated_at": "Not updated yet"
    }

    try:
        with open(STATUS_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return default_status
    except json.JSONDecodeError:
        return default_status


def format_last_msg(last_msg):
    if isinstance(last_msg, dict):
        device_id = last_msg.get("device_id", "unknown")
        temp = last_msg.get("temp", "N/A")
        humidity = last_msg.get("humidity", "N/A")
        return f"{device_id} | temp={temp} | humidity={humidity}"

    return str(last_msg)


def show_monitor(status):
    devices = status.get("devices_online", [])
    device_count = len(devices) if isinstance(devices, list) else devices

    print("+--------------------------------------------------+")
    print("| PRIVATE SHIELD v0.1                              |")
    print("+--------------------------------------------------+")
    print(f"| Running: {str(status.get('running', False)):<39} |")
    print(f"| Devices online: {str(device_count):<31} |")
    print(f"| Total messages: {str(status.get('total_messages', 0)):<31} |")
    print(f"| Msgs last 10s: {str(status.get('messages_last_10s', 0)):<31} |")
    print(f"| Alerts fired: {str(status.get('alerts_fired', 0)):<32} |")
    print(f"| Last msg: {format_last_msg(status.get('last_msg', {})):<37} |")
    print(f"| Updated at: {str(status.get('updated_at', 'N/A')):<35} |")
    print("+--------------------------------------------------+")


def main():
    while True:
        clear_screen()
        status = load_status()
        show_monitor(status)
        time.sleep(2)


if __name__ == "__main__":
    main()