# Import the standard system library to handle command-line arguments and clean script exiting
import sys
# Import the operating system library to create directories and manage file paths
import os
# Import the time library to calculate time differences, keep history, and pause execution
import time
# Import the json library to parse incoming JSON payloads and format the status file
import json
# Import the datetime module to generate human-readable timestamps for our logged rows
from datetime import datetime
# Import pandas to easily write collected lists of dictionary rows to the CSV dataset
import pandas as pd
# Import the MQTT client library to connect to the broker and receive published IoT messages
import paho.mqtt.client as mqtt

# Configuration variables: Broker host, broker port, and the topic we subscribe to
BROKER_HOST = "localhost" # Connect to localhost by default
BROKER_PORT = 1883        # Connect to port 1883 by default
TOPIC = "devices/#"       # Subscribe to the topic devices/#

# Define the folder path and file paths for our CSV and status files
DATA_DIR = "data"                                         # Directory where data will be stored
CSV_PATH = os.path.join(DATA_DIR, "normal.csv")           # File path to save normal IoT traffic
STATUS_PATH = os.path.join(DATA_DIR, "status.json")       # File path to write live status info for monitor.py

# Create the data directory if it does not already exist
os.makedirs(DATA_DIR, exist_ok=True)                      # Create folder recursively; do nothing if it exists

# Core state variables to track progress and statistics
total_messages_count = 0  # Running counter of all valid messages received in this session
message_buffer = []       # Temporary in-memory list to buffer rows before writing to CSV
message_history = []      # List of dictionaries tracking (time, device_id) to calculate 10s rate
last_message_data = "N/A" # Holds the payload of the last received message to update status.json

# Function to save all buffered rows to the CSV file
def flush_buffer_to_csv():
    global message_buffer # Use the global message buffer list
    if not message_buffer: # If the buffer is empty, do nothing
        return            # Exit the function early
    
    # Convert the buffered dictionaries into a pandas DataFrame
    df = pd.DataFrame(message_buffer)
    # Check if the CSV file already exists on disk
    file_exists = os.path.exists(CSV_PATH)
    # Append the DataFrame to the CSV; write headers only if the file does not exist yet
    df.to_csv(CSV_PATH, mode='a', header=not file_exists, index=False)
    # Clear the in-memory buffer list so we do not write duplicate rows next time
    message_buffer.clear()

# Function to write the current status metrics to data/status.json
def update_status_json(running=True):
    # Prune history to keep only messages from the last 10 seconds
    current_time = time.time() # Get the current epoch time in seconds
    # Filter the message history list to keep messages newer than 10 seconds ago
    recent_messages = [m for m in message_history if m["time"] > current_time - 10]
    
    # Calculate the number of messages received in the last 10 seconds
    messages_last_10s = len(recent_messages)
    # Extract unique device IDs from the recent messages to list online devices
    devices_online = list(set(m["device_id"] for m in recent_messages))
    
    # Construct the dictionary representing the current state
    status_data = {
        "running": running,                       # True if the logger is running, False if stopped
        "topic": TOPIC,                           # The MQTT topic we are subscribed to
        "normal_csv": CSV_PATH,                   # The path to the CSV file where data is logged
        "total_messages": total_messages_count,   # Total messages processed during this run
        "messages_last_10s": messages_last_10s,   # Count of messages in the last 10 seconds
        "devices_online": devices_online,         # List of unique devices online in the last 10s
        "alerts_fired": 0,                        # Always 0 because this is normal traffic (no anomalies)
        "last_msg": last_message_data,            # The dictionary payload of the last received message
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Current time formatted as string
    }
    
    # Write the dictionary to status.json using clean indenting
    try:
        with open(STATUS_PATH, "w") as f:          # Open status.json for writing
            json.dump(status_data, f, indent=4)    # Serialize the dictionary to JSON format
    except Exception as e:                         # Catch any file system writing errors
        print(f"Error updating status.json: {e}") # Print the error to standard output

# Callback function executed when the client successfully connects to the MQTT broker
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:                                                 # Return code 0 means connection succeeded
        print(f"Connected successfully to broker at {BROKER_HOST}:{BROKER_PORT}") # Print success message
        client.subscribe(TOPIC)                                 # Subscribe to the devices/# topic
        print(f"Subscribed to topic: {TOPIC}")                  # Log subscription info
    else:                                                       # Any other return code means failure
        print(f"Connection failed with code {rc}")              # Print connection failure code

# Callback function executed when a message is received from the subscribed MQTT topic
def on_message(client, userdata, msg):
    global total_messages_count, message_buffer, message_history, last_message_data # Access global state variables
    
    try:
        # Decode the byte payload into a UTF-8 string
        payload_str = msg.payload.decode('utf-8')
        # Parse the JSON string into a Python dictionary
        payload_dict = json.loads(payload_str)
    except Exception as e:                                      # Handle JSON decode errors gracefully
        print(f"Skipping invalid JSON message on topic {msg.topic}: {e}") # Print failure message
        return                                                  # Exit callback early without logging
    
    # Record current timestamp for message rate tracking
    current_time = time.time()                                  # Get epoch time in seconds
    
    # Extract the device ID from the payload dictionary, falling back to topic split if missing
    device_id = payload_dict.get("device_id")                   # Read device_id key
    if not device_id:                                           # If device_id key is missing
        topic_parts = msg.topic.split('/')                      # Split topic string by slash
        device_id = topic_parts[1] if len(topic_parts) > 1 else "unknown" # Use second part or unknown
    
    # Append this message's timestamp and device ID to our running history list
    message_history.append({"time": current_time, "device_id": device_id})
    
    # Keep track of the last message dictionary for status updates
    last_message_data = payload_dict                            # Store parsed payload dictionary
    
    # Create the row dictionary to be saved in our CSV file
    row_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Current timestamp formatted
        "topic": msg.topic,                                      # The exact topic the message arrived on
        **payload_dict,                                          # Unpack all keys/values from the JSON payload
        "label": 0                                               # Append label = 0 for normal traffic
    }
    
    # Add the row to our in-memory buffer list
    message_buffer.append(row_data)                             # Append row dictionary to list
    
    # Increment the running counter of collected messages
    total_messages_count += 1                                   # Increment count by 1
    
    # Print progress information every 50 messages
    if total_messages_count % 50 == 0:                          # Check if count is a multiple of 50
        print(f"Progress: Collected {total_messages_count} messages so far...") # Log progress count
        flush_buffer_to_csv()                                   # Write current buffer to file immediately
    
    # Write to CSV immediately if the buffer exceeds a small batch size of 10
    if len(message_buffer) >= 10:                               # Check if buffer has 10 or more rows
        flush_buffer_to_csv()                                   # Write and flush the buffer

# Main program block
if __name__ == "__main__":
    # Create an MQTT client instance, handling CallbackAPIVersion for paho-mqtt v2.x
    try:
        # Try initializing with CallbackAPIVersion for Paho-MQTT v2.x compatibility
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # Initialize MQTT Client V2
    except AttributeError:
        # Fallback to Paho-MQTT v1.x initialization if version 2 is not present
        client = mqtt.Client()                                   # Initialize MQTT Client V1
        
    # Assign the connection and message handlers
    client.on_connect = on_connect                              # Bind connection callback function
    client.on_message = on_message                              # Bind message callback function
    
    # Connect to the local Mosquitto MQTT broker
    print(f"Connecting to MQTT Broker at {BROKER_HOST}:{BROKER_PORT}...") # Print connection message
    try:
        client.connect(BROKER_HOST, BROKER_PORT, 60)            # Connect to broker with 60s timeout
    except Exception as e:                                      # Catch broker connection failure
        print("\n" + "="*50)                                    # Visual separator line
        print("ERROR: Could not connect to the MQTT Broker!")   # Print main error statement
        print(f"Details: {e}")                                  # Print the actual traceback/exception message
        print("Please ensure that your Mosquitto MQTT Broker is running on localhost:1883.") # Give actionable tip
        print("="*50 + "\n")                                    # Visual separator line
        sys.exit(1)                                             # Exit script with error status code 1
        
    # Start the network loop in a background thread to process incoming messages
    client.loop_start()                                         # Start background network loop thread
    
    # Write initial status to status.json indicating the logger is active
    update_status_json(running=True)                            # Update status.json with running=True
    
    print("MQTT Logger is running. Collecting normal dataset. Press Ctrl+C to stop.") # Alert user
    
    # Main execution loop: Run until we reach at least 500 messages
    try:
        while total_messages_count < 500:                       # Check if total messages is less than 500
            time.sleep(1)                                       # Sleep for 1 second to reduce CPU usage
            update_status_json(running=True)                    # Periodically update status.json to keep rate accurate
            
        print(f"\nTarget achieved! Collected {total_messages_count} messages.") # Log target reached
        
    except KeyboardInterrupt:                                   # Catch Ctrl+C interruption from terminal
        print("\nCtrl+C detected! Safely shutting down...")     # Print shutdown alert
        
    finally:                                                    # Run this clean up block always on exit
        # Write any remaining buffered rows in memory to normal.csv
        print("Saving remaining buffered rows to CSV...")       # Log buffer flush step
        flush_buffer_to_csv()                                   # Flush remaining rows to file
        
        # Stop the background network loop thread
        client.loop_stop()                                      # Terminate the background MQTT thread
        # Disconnect from the broker
        client.disconnect()                                     # Cleanly disconnect network client socket
        
        # Write final status to status.json indicating the logger is stopped
        update_status_json(running=False)                       # Update status.json with running=False
        
        # Print final summaries if we have logged messages and normal.csv exists
        if os.path.exists(CSV_PATH):                            # Check if the normal.csv file exists
            try:
                # Load the collected data using pandas
                df = pd.read_csv(CSV_PATH)                      # Read CSV file into pandas DataFrame
                print("\n" + "="*50)                            # Print separator line
                print(f"Collected {len(df)} rows in total.")     # Print count of total rows collected
                print("\nLast few rows of the CSV:")            # Header for tail print
                print(df.tail(5))                               # Print last 5 rows of the DataFrame
                print("\nCSV Summary Statistics:")              # Header for describe print
                print(df.describe(include="all"))               # Print standard descriptive stats
                print("="*50)                                   # Print separator line
            except Exception as e:                              # Catch errors while loading or printing statistics
                print(f"Error generating final dataset report: {e}") # Print the reporting error
        else:                                                   # If normal.csv does not exist on exit
            print("No CSV data file found to report on.")       # Print notice statement
