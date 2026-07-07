# PRIVATE SHIELD - Week 2 Person C

Person C deliverable: `attack_simulator.py`.

## Work Summary

Person C builds the attack traffic generator for the project. In Week 2, the goal is to simulate three local attack patterns and measure their rate:

| Mode | What it simulates | Output metric |
| --- | --- | --- |
| DoS | Floods the MQTT broker with many messages per second | MQTT messages/second |
| Port Scan | Checks ports 1 to 1000 to find open services | connection attempts/second |
| Mirai | Opens 50 simultaneous TCP connections like a botnet burst | connections/second |

## Install

```bash
pip install -r requirements.txt
```

For DoS and Mirai mode, keep Mosquitto running:

```bash
mosquitto -p 1883
```

## Run Commands

DoS attack for 30 seconds:

```bash
python attack_simulator.py --mode dos --duration 30 --rate 500
```

Port scan for 30 seconds:

```bash
python attack_simulator.py --mode port_scan --duration 30
```

Mirai-style burst for 30 seconds:

```bash
python attack_simulator.py --mode mirai --duration 30 --connections 50
```

## How To Explain To Faculty

In Week 2, my responsibility was the security and attack simulation part of PRIVATE SHIELD. I implemented `attack_simulator.py`, which creates controlled attack traffic on localhost so that later our logger, dataset builder, and ML detector can learn the difference between normal IoT behavior and malicious behavior.

Bangla + English explanation:

Amar kaj holo project-er attack simulation layer build kora. Normal fake IoT devices jevabe MQTT broker-e message pathay, attacker script-o same local environment-e suspicious traffic generate kore. DoS mode broker-e high rate-e MQTT message publish kore, Port Scan mode local machine-er port 1 theke 1000 porjonto check kore kon port open ache, and Mirai mode ekshathe 50 ta TCP connection open kore botnet-style burst behavior simulate kore. Ei output theke amra messages-per-second / connections-per-second measure korte pari, ja Week 3-e labelled dataset banate help korbe.

Important point: This is only a safe local lab simulation using `localhost`. It is not designed for attacking any external system.
