import socket
import threading
import time

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 1883

WORKER_THREADS = 5
TEST_DURATION_SECONDS = 20
CONNECTION_HOLD_SECONDS = 0.2
DELAY_BETWEEN_ATTEMPTS = 0.1

success_count = 0
failure_count = 0
lock = threading.Lock()
stop_event = threading.Event()


def test_connection():
    global success_count, failure_count

    while not stop_event.is_set():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((TARGET_HOST, TARGET_PORT))

                with lock:
                    success_count += 1

                time.sleep(CONNECTION_HOLD_SECONDS)

        except Exception:
            with lock:
                failure_count += 1

        time.sleep(DELAY_BETWEEN_ATTEMPTS)


def main():
    print("Starting controlled localhost resilience test...")
    print(f"Target: {TARGET_HOST}:{TARGET_PORT}")
    print(f"Threads: {WORKER_THREADS}")
    print(f"Duration: {TEST_DURATION_SECONDS} seconds")
    print("Press Ctrl+C to stop early.\n")

    threads = []

    for _ in range(WORKER_THREADS):
        t = threading.Thread(target=test_connection)
        t.start()
        threads.append(t)

    start_time = time.time()

    try:
        while time.time() - start_time < TEST_DURATION_SECONDS:
            time.sleep(1)

            with lock:
                current_success = success_count
                current_failure = failure_count

            elapsed = int(time.time() - start_time)

            print(
                f"[{elapsed}s] Successful connections: {current_success} | "
                f"Failed attempts: {current_failure}"
            )

    except KeyboardInterrupt:
        print("\nTest stopped by user.")

    finally:
        stop_event.set()

        for t in threads:
            t.join()

        print("\nFinal Result:")
        print(f"Total successful connections: {success_count}")
        print(f"Total failed attempts: {failure_count}")
        print("Controlled localhost test completed.")


if __name__ == "__main__":
    main()
