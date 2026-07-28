# WEF Research Paper Summary

## Research Problem

This research focuses on detecting attacks and abnormal activities in IoT network traffic.

## Proposed Solution

The proposed system uses federated learning. In federated learning, different clients train their models using their own local data.

The clients do not directly share their private raw data. They only send model updates to a central server.

## Main Objectives

- Detect abnormal IoT network traffic
- Identify attack-related activities
- Protect the privacy of client data
- Train a common global model
- Improve attack detection performance

## Input

The input of the system is IoT network traffic data collected from different devices or clients.

## Output

The system predicts whether the network traffic is normal or abnormal.

## My Understanding

Each client keeps its dataset locally and trains a local model. After training, the client sends its model update to the central server.

The central server combines the updates from all clients and creates a global model.
