from pathlib import Path
import sys

import numpy as np
import torch


# ---------------- PATHS ----------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Allow test.py to use the original FedMSE code inside src
sys.path.insert(0, str(SRC))

from Model.Shrink_Autoencoder import Shrink_Autoencoder
from Model.Centroid import CentroidBasedOneClassClassifier
from DataLoader.dataloader import load_data, IoTDataProccessor


# ---------------- FILE LOCATIONS ----------------

MODEL_PATH = ROOT / "Outputs" / "global_model.pt"

CLIENT_PATH = (
    ROOT
    / "Data"
    / "N-BaIoT"
    / "IID-10-Client_Data"
    / "Client-1"
)

NORMAL_PATH = CLIENT_PATH / "normal"
ATTACK_PATH = CLIENT_PATH / "abnormal"
TEST_NORMAL_PATH = CLIENT_PATH / "test_normal"


# ---------------- LOAD DATA ----------------

print("\nFedMSE Attack Detection Test")
print("=" * 35)

normal_data = load_data(str(NORMAL_PATH))

# Shuffle normal data in a repeatable way
normal_data = normal_data.sample(
    frac=1,
    random_state=1234
).reset_index(drop=True)

# Use 40% normal data like the FedMSE training setup
train_size = int(0.4 * len(normal_data))
train_normal = normal_data.iloc[:train_size]

test_normal = load_data(str(TEST_NORMAL_PATH))
attack_data = load_data(str(ATTACK_PATH))


# ---------------- SCALE DATA ----------------

processor = IoTDataProccessor(scaler="standard")

train_x, _ = processor.fit_transform(train_normal)

normal_x, _ = processor.transform(test_normal)

attack_x, _ = processor.transform(
    attack_data,
    type="abnormal"
)

# Keep demo fast
normal_x = normal_x[:200]
attack_x = attack_x[:200]


# ---------------- LOAD TRAINED MODEL ----------------

model = Shrink_Autoencoder(
    input_dim=115,
    output_dim=115,
    hidden_neus=50,
    latent_dim=11,
    shrink_lambda=10
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ---------------- CREATE LATENT FEATURES ----------------

def get_latent(data):

    tensor_data = torch.tensor(
        data,
        dtype=torch.float32
    )

    with torch.no_grad():

        latent, _, _ = model(tensor_data)

    return latent.cpu().numpy()


train_latent = get_latent(train_x)

normal_latent = get_latent(normal_x)

attack_latent = get_latent(attack_x)


# ---------------- CENTROID DETECTOR ----------------

# 95% of normal-training behaviour is treated as normal.
# Far-away data is treated as suspicious.
detector = CentroidBasedOneClassClassifier(
    threshold=0.95
)

detector.fit(train_latent)

normal_prediction = detector.predict(normal_latent)

attack_prediction = detector.predict(attack_latent)


# ---------------- RESULTS ----------------

normal_correct = np.mean(
    normal_prediction == False
) * 100

attack_detected = np.mean(
    attack_prediction == True
) * 100


print("\nNORMAL TRAFFIC")
print(f"Normal samples checked: {len(normal_prediction)}")
print(f"Correctly accepted as normal: {normal_correct:.2f}%")

if normal_correct >= 50:
    print("Result: NORMAL")
else:
    print("Result: SUSPICIOUS")


print("\nATTACK TRAFFIC")
print(f"Attack samples checked: {len(attack_prediction)}")
print(f"Attacks detected: {attack_detected:.2f}%")

if attack_detected >= 50:
    print("Result: ATTACK DETECTED")
else:
    print("Result: ATTACK MISSED")


print("\nFedMSE test completed.")