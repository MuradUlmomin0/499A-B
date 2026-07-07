"""Week 2 / Person C attack simulator for PRIVATE SHIELD.

Runs three local-only security test modes against the MQTT lab broker:

1. DoS       - publish many MQTT messages per second.
2. Port scan - check which localhost ports from 1..1000 accept TCP.
3. Mirai     - open connection waves with many clients at the same time.

This script is for controlled CSE499 lab testing on your own localhost broker.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_HOST = "localhost"
DEFAULT_MQTT_PORT = 1883
DEFAULT_DURATION = 30


@dataclass
class AttackResult:
    mode: str
    duration_seconds: float
    total_events: int
    rate_per_second: float
    details: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mqtt_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(2, "big") + encoded


def encode_remaining_length(length: int) -> bytes:
    encoded = bytearray()
    while True:
        digit = length % 128
        length //= 128
        if length > 0:
            digit |= 128
        encoded.append(digit)
        if length == 0:
            return bytes(encoded)


class RawMqttPublisher:
    """Minimal MQTT 3.1.1 QoS 0 publisher using only sockets."""

    def __init__(self, host: str, port: int, client_id: str) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.sock: socket.socket | None = None

    def __enter__(self) -> "RawMqttPublisher":
        self.sock = socket.create_connection((self.host, self.port), timeout=5)

        variable_header = mqtt_string("MQTT") + bytes([4, 2]) + (60).to_bytes(2, "big")
        payload = mqtt_string(self.client_id)
        remaining_length = len(variable_header) + len(payload)
        packet = bytes([0x10]) + encode_remaining_length(remaining_length) + variable_header + payload

        self.sock.sendall(packet)
        connack = self.sock.recv(4)
        if len(connack) < 4 or connack[0] != 0x20 or connack[3] != 0:
            raise ConnectionRefusedError(f"MQTT broker rejected connection: {connack!r}")
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.sock is not None:
            try:
                self.sock.sendall(bytes([0xE0, 0x00]))
            finally:
                self.sock.close()
                self.sock = None

    def publish(self, topic: str, payload: str) -> None:
        if self.sock is None:
            raise RuntimeError("MQTT publisher is not connected")

        body = mqtt_string(topic) + payload.encode("utf-8")
        packet = bytes([0x30]) + encode_remaining_length(len(body)) + body
        self.sock.sendall(packet)


def make_attack_payload(sequence: int, rate: int) -> str:
    payload = {
        "device_id": "attacker_dos_01",
        "device_type": "attacker",
        "attack_type": "dos",
        "temperature": 0,
        "humidity": 0,
        "packets_per_sec": rate,
        "bytes_per_pkt": 1400,
        "port": DEFAULT_MQTT_PORT,
        "sequence": sequence,
        "generated_at": utc_now(),
    }
    return json.dumps(payload)


def run_dos(host: str, port: int, duration: int, rate: int, topic: str) -> AttackResult:
    """Publish MQTT flood messages at a controlled target rate."""
    start = time.perf_counter()
    end = start + duration
    sent = 0
    next_publish = start
    interval = 1 / max(1, rate)

    with RawMqttPublisher(host, port, "private_shield_dos_attacker") as publisher:
        while time.perf_counter() < end:
            now = time.perf_counter()
            if now < next_publish:
                time.sleep(min(next_publish - now, 0.01))
                continue

            publisher.publish(topic, make_attack_payload(sent + 1, rate))
            sent += 1
            next_publish += interval

    elapsed = max(time.perf_counter() - start, 0.001)
    return AttackResult(
        mode="dos",
        duration_seconds=elapsed,
        total_events=sent,
        rate_per_second=sent / elapsed,
        details={"topic": topic, "target_rate": rate},
    )


def try_connect(host: str, port: int, timeout: float) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def run_port_scan(
    host: str,
    duration: int,
    start_port: int,
    end_port: int,
    timeout: float,
) -> AttackResult:
    """Scan a local port range repeatedly until the duration ends."""
    scan_start = time.perf_counter()
    scan_end = scan_start + duration
    attempts = 0
    rounds = 0
    open_ports: set[int] = set()

    while time.perf_counter() < scan_end:
        rounds += 1
        for port in range(start_port, end_port + 1):
            if time.perf_counter() >= scan_end:
                break
            attempts += 1
            if try_connect(host, port, timeout):
                open_ports.add(port)

    elapsed = max(time.perf_counter() - scan_start, 0.001)
    return AttackResult(
        mode="port_scan",
        duration_seconds=elapsed,
        total_events=attempts,
        rate_per_second=attempts / elapsed,
        details={
            "ports_checked": f"{start_port}-{end_port}",
            "rounds_completed": rounds,
            "open_ports": sorted(open_ports),
        },
    )


def mirai_worker(
    host: str,
    port: int,
    timeout: float,
    barrier: threading.Barrier,
    results: list[bool],
    index: int,
) -> None:
    try:
        barrier.wait()
        results[index] = try_connect(host, port, timeout)
    except Exception:
        results[index] = False


def run_mirai(host: str, port: int, duration: int, connections: int, timeout: float) -> AttackResult:
    """Open repeated waves of simultaneous TCP connections."""
    start = time.perf_counter()
    end = start + duration
    total_attempts = 0
    successful = 0
    waves = 0

    while time.perf_counter() < end:
        waves += 1
        results = [False] * connections
        barrier = threading.Barrier(connections)
        threads = [
            threading.Thread(
                target=mirai_worker,
                args=(host, port, timeout, barrier, results, index),
                daemon=True,
            )
            for index in range(connections)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        total_attempts += connections
        successful += sum(1 for ok in results if ok)

    elapsed = max(time.perf_counter() - start, 0.001)
    return AttackResult(
        mode="mirai",
        duration_seconds=elapsed,
        total_events=total_attempts,
        rate_per_second=total_attempts / elapsed,
        details={
            "connections_per_wave": connections,
            "waves": waves,
            "successful_connections": successful,
        },
    )


def print_result(result: AttackResult) -> None:
    print("\nPRIVATE SHIELD - Person C Week 2 Attack Result")
    print(f"Mode: {result.mode}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")
    print(f"Total events: {result.total_events}")
    print(f"Rate: {result.rate_per_second:.2f} events/second")
    print("Details:")
    for key, value in result.details.items():
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 2 Person C local attack simulations for PRIVATE SHIELD."
    )
    parser.add_argument(
        "--mode",
        choices=("dos", "port_scan", "mirai"),
        required=True,
        help="Attack mode to run.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Target host. Keep localhost for lab use.")
    parser.add_argument("--port", default=DEFAULT_MQTT_PORT, type=int, help="Target MQTT/TCP port.")
    parser.add_argument("--duration", default=DEFAULT_DURATION, type=int, help="Run duration in seconds.")
    parser.add_argument("--rate", default=500, type=int, help="DoS MQTT messages per second.")
    parser.add_argument("--topic", default="devices/attack/dos", help="MQTT topic for DoS messages.")
    parser.add_argument("--start-port", default=1, type=int, help="Port scan start port.")
    parser.add_argument("--end-port", default=1000, type=int, help="Port scan end port.")
    parser.add_argument("--connections", default=50, type=int, help="Mirai simultaneous connections.")
    parser.add_argument("--timeout", default=0.05, type=float, help="TCP connection timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.duration <= 0:
        print("--duration must be greater than 0")
        return 2

    print("Starting local attack simulation...")
    print(f"Target: {args.host}:{args.port}")
    print(f"Mode: {args.mode}")

    try:
        if args.mode == "dos":
            result = run_dos(args.host, args.port, args.duration, args.rate, args.topic)
        elif args.mode == "port_scan":
            result = run_port_scan(
                args.host,
                args.duration,
                args.start_port,
                args.end_port,
                args.timeout,
            )
        else:
            result = run_mirai(args.host, args.port, args.duration, args.connections, args.timeout)
    except ConnectionRefusedError:
        print("Could not connect. Start Mosquitto first, then run this script again.")
        return 1
    except KeyboardInterrupt:
        print("\nAttack simulation stopped by user.")
        return 130

    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
