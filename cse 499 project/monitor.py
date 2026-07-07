import os
import time
import json
from datetime import datetime

# Define file paths
DATA_DIR = "data"
COUNTER_FILE = os.path.join(DATA_DIR, "counter.json")
STATUS_FILE = os.path.join(DATA_DIR, "status.json")

def initialize_shared_data():
    """Initializes the shared JSON file if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(STATUS_FILE) and not os.path.exists(COUNTER_FILE):
        default_data = {
            "msg_count": 0,
            "alerts": [],
            "last_msg_time": datetime.now().strftime("%H:%M:%S")
        }
        with open(COUNTER_FILE, "w") as f:
            json.dump(default_data, f, indent=4)
        print(f"Initialized shared file: {COUNTER_FILE}")

def read_shared_data():
    """Reads and returns the data from status.json (preferred) or counter.json."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                data = json.load(f)
                return {
                    "is_status_json": True,
                    "running": data.get("running", False),
                    "devices_online": data.get("devices_online", []),
                    "messages_last_10s": data.get("messages_last_10s", 0),
                    "total_messages": data.get("total_messages", 0),
                    "alerts_fired": data.get("alerts_fired", 0),
                    "last_msg": data.get("last_msg", "N/A"),
                    "updated_at": data.get("updated_at", "N/A")
                }
        except Exception:
            pass # Fallback to counter.json if status.json is corrupt or unreadable

    try:
        with open(COUNTER_FILE, "r") as f:
            data = json.load(f)
            return {
                "is_status_json": False,
                "msg_count": data.get("msg_count", 0),
                "alerts": data.get("alerts", []),
                "last_msg_time": data.get("last_msg_time", "N/A")
            }
    except Exception:
        return {
            "is_status_json": False,
            "msg_count": 0,
            "alerts": [],
            "last_msg_time": "N/A"
        }

def format_box_line(content):
    """Formats a single line to fit exactly inside the ASCII box border."""
    # The box interior is exactly 29 characters wide.
    return f"│ {content.ljust(27)} │"

def display_dashboard(data):
    """Prints the dashboard inside the exact requested ASCII box layout."""
    if data.get("is_status_json", False):
        devices_online_data = data.get("devices_online", [])
        devices_count = len(devices_online_data) if isinstance(devices_online_data, list) else devices_online_data
        msgs_count = data.get("messages_last_10s", 0)
        alerts_fired = data.get("alerts_fired", 0)
        
        last_msg = data.get("last_msg", "N/A")
        if isinstance(last_msg, dict):
            dev_id = last_msg.get("device_id", "cam_01")
            timestamp_part = data.get("updated_at", "N/A").split(" ")[-1]
            last_msg_str = f"{dev_id} @ {timestamp_part}"
        else:
            last_msg_str = "cam_01 @ N/A"
            
        print("┌─────────────────────────────┐")
        status_text = " ACTIVE" if data.get("running", False) else " INACTIVE"
        print(format_box_line(f" PRIVATE SHIELD ({status_text})"))
        print(format_box_line(f" Devices online:  {devices_count}"))
        print(format_box_line(f" Msgs last 10s:   {msgs_count}"))
        print(format_box_line(f" Alerts fired:    {alerts_fired}"))
        print(format_box_line(f" Last msg: {last_msg_str}"))
        print("└─────────────────────────────┘")
    else:
        msgs_count = data.get("msg_count", 0)
        alerts_fired = len(data.get("alerts", []))
        last_msg_time = data.get("last_msg_time", "N/A")
        
        print("┌─────────────────────────────┐")
        print(format_box_line(" PRIVATE SHIELD v0.1"))
        print(format_box_line(" Devices online:  3"))
        print(format_box_line(f" Msgs last 10s:   {msgs_count}"))
        print(format_box_line(f" Alerts fired:    {alerts_fired}"))
        print(format_box_line(f" Last msg: cam_01 @ {last_msg_time}"))
        print("└─────────────────────────────┘")

def main():
    initialize_shared_data()
    
    while True:
        # Clear the console screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Read the current shared data
        data = read_shared_data()
        
        # Display the ASCII box dashboard
        display_dashboard(data)
        
        time.sleep(2)

if __name__ == "__main__":
    main()
