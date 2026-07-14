import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "data/labelled.csv"

# Since we are inside "cse 499 project",
# top-level Others/images folder is one step back.
OUTPUT_DIR = "../Others/images"

HISTOGRAM_FILE = os.path.join(OUTPUT_DIR, "week3_packets_histogram.png")
SCATTER_FILE = os.path.join(OUTPUT_DIR, "week3_bytes_flow_scatter.png")


def label_name(label):
    label = int(label)

    labels = {
        0: "Normal",
        1: "DoS",
        2: "Port Scan",
        3: "Mirai"
    }

    return labels.get(label, f"Unknown-{label}")


def check_required_columns(df):
    required_columns = [
        "packets_per_sec",
        "bytes_per_pkt",
        "flow_duration",
        "label"
    ]

    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return False

    return True


def create_packets_histogram(df):
    plt.figure(figsize=(8, 5))

    for label in sorted(df["label"].unique()):
        label_data = df[df["label"] == label]

        plt.hist(
            label_data["packets_per_sec"],
            alpha=0.6,
            label=label_name(label)
        )

    plt.title("Packets Per Second by Traffic Type")
    plt.xlabel("Packets Per Second")
    plt.ylabel("Number of Records")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTOGRAM_FILE)
    plt.close()

    print(f"Saved histogram: {HISTOGRAM_FILE}")


def create_bytes_flow_scatter(df):
    plt.figure(figsize=(8, 5))

    for label in sorted(df["label"].unique()):
        label_data = df[df["label"] == label]

        plt.scatter(
            label_data["bytes_per_pkt"],
            label_data["flow_duration"],
            alpha=0.7,
            label=label_name(label)
        )

    plt.title("Bytes Per Packet vs Flow Duration")
    plt.xlabel("Bytes Per Packet")
    plt.ylabel("Flow Duration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCATTER_FILE)
    plt.close()

    print(f"Saved scatter plot: {SCATTER_FILE}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if not check_required_columns(df):
        return

    print("Label counts:")
    print(df["label"].value_counts())

    create_packets_histogram(df)
    create_bytes_flow_scatter(df)

    print("Week 3 visualisation completed.")


if __name__ == "__main__":
    main()