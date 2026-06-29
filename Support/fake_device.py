import time       # Allows us to pause the script (sleep)
import random     # Generates the fake temperature and humidity numbers
import json       # Converts our Python dictionary into a text format computers can send over networks
import sys        # Allows the script to read commands typed in the terminal
import paho.mqtt.client as mqtt # The core library that handles the MQTT network protocol

# CONFIGURATION SECTION: Setting up the device identity
# This checks if we typed a name in the terminal (like 'cam_01'). 
# If we didn't, it defaults to 'sensor_generic'.
device_id = sys.argv[1] if len(sys.argv) > 1 else "sensor_generic"

# NETWORK SETUP: Where is the server?
MQTT_BROKER = "127.0.0.1"               # The IP address of the local machine (localhost)
MQTT_PORT = 1883                        # The default port number Mosquitto listens on
MQTT_TOPIC = f"devices/{device_id}"     # The 'channel' this specific device will broadcast on

# INITIALIZATION: Create the network agent
# (Note: Using CallbackAPIVersion.VERSION2 prevents that deprecation warning!)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# CONNECTION BLOCK: Try to connect to the broker
try:
    # Connect to the broker. The '60' means it will timeout if the broker doesn't respond in 60 seconds.
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print(f"[{device_id}] Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
except Exception as e:
    # If the broker is off, print the error and kill the script so it doesn't crash wildly.
    print(f"[{device_id}] Failed to connect: {e}")
    sys.exit(1)

# THE MAIN LOOP: Run forever until the user hits Ctrl+C
while True:
    # 1. GENERATE DATA: Create a dictionary simulating structural packet profiles
    payload = {
        "device_id": device_id,
        "temp": round(random.uniform(20.0, 35.0), 2),      # Random decimal between 20-35
        "humidity": round(random.uniform(40.0, 80.0), 2),  # Random decimal between 40-80
        "packets_per_sec": 1,
        "bytes_per_pkt": 64,
        "port": MQTT_PORT
    }
    
    # 2. SERIALIZE: Convert the Python dictionary into a standard JSON string
    json_payload = json.dumps(payload)
    
    # 3. TRANSMIT: Send the JSON string to the broker on this device's specific topic channel
    client.publish(MQTT_TOPIC, json_payload)
    
    # 4. LOGGING: Print it to our own screen so we know it worked
    print(f"[{device_id}] Published: {json_payload}")
    
    # 5. REST: Pause the script for 3 seconds before generating the next packet
    time.sleep(3)