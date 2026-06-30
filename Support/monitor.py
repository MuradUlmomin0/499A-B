import os
import time
import json
from datetime import datetime

# Define file paths
DATA_DIR = "data"
COUNTER_FILE = os.path.join(DATA_DIR, "counter.json")

def initialize_shared_data():
    """Initializes the shared JSON file if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(COUNTER_FILE):
        default_data = {
            "msg_count": 0,
            "alerts": [],
            "last_msg_time": datetime.now().strftime("%H:%M:%S")
        }
        with open(COUNTER_FILE, "w") as f:
            json.dump(default_data, f, indent=4)
        print(f"Initialized shared file: {COUNTER_FILE}")

def read_shared_data():
    """Reads and returns the data from the shared JSON file."""
    try:
        with open(COUNTER_FILE, "r") as f:
            return json.load(f)
    except Exception:
        # Fallback if read fails or file is empty/corrupt
        return {
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
    msgs_count = data.get("msg_count", 0)
    alerts_fired = len(data.get("alerts", []))
    last_msg_time = data.get("last_msg_time", "N/A")
    
    # Visual ASCII box components
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
