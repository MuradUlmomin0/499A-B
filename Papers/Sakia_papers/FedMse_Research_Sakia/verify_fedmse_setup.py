import importlib
import json
import sys
from pathlib import Path


# Current FedMSE project folder
PROJECT_ROOT = Path(__file__).resolve().parent

# Location of the prepared 10-client dataset
DATASET_PATH = (
    PROJECT_ROOT
    / "Data"
    / "N-BaIoT"
    / "IID-10-Client_Data"
)

# Location of the official FedMSE 10-client configuration
CONFIG_PATH = (
    PROJECT_ROOT
    / "src"
    / "Configuration"
    / "scen2-nba-iot-10clients.json"
)

# Folder where verification results will be saved
OUTPUT_PATH = PROJECT_ROOT / "Outputs"

# Required folders inside every client
REQUIRED_DATA_FOLDERS = [
    "normal",
    "abnormal",
    "test_normal",
]

# Package name and Python import name
REQUIRED_PACKAGES = {
    "Pandas": "pandas",
    "NumPy": "numpy",
    "PyTorch": "torch",
    "Scikit-learn": "sklearn",
    "Matplotlib": "matplotlib",
    "tqdm": "tqdm",
}


def count_files(folder):
    """Count all files inside a folder."""

    if not folder.exists():
        return 0

    return sum(
        1
        for item in folder.rglob("*")
        if item.is_file()
    )


def check_dependencies():
    """Check whether required Python libraries are installed."""

    results = {}

    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")

            results[package_name] = {
                "installed": True,
                "version": str(version),
            }

        except ImportError:
            results[package_name] = {
                "installed": False,
                "version": None,
            }

    return results


def check_configuration():
    """Check whether the configuration file exists and contains valid JSON."""

    result = {
        "path": str(CONFIG_PATH),
        "exists": CONFIG_PATH.exists(),
        "valid_json": False,
    }

    if not CONFIG_PATH.exists():
        return result

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            json.load(file)

        result["valid_json"] = True

    except (OSError, json.JSONDecodeError) as error:
        result["error"] = str(error)

    return result


def check_clients():
    """Check Client-1 to Client-10 and their data folders."""

    client_results = []

    for client_number in range(1, 11):
        client_name = f"Client-{client_number}"
        client_path = DATASET_PATH / client_name

        client_information = {
            "client": client_name,
            "exists": client_path.exists(),
            "folders": {},
        }

        for folder_name in REQUIRED_DATA_FOLDERS:
            folder_path = client_path / folder_name
            file_count = count_files(folder_path)

            client_information["folders"][folder_name] = {
                "exists": folder_path.exists(),
                "file_count": file_count,
            }

        client_results.append(client_information)

    return client_results


def create_text_report(summary):
    """Convert the verification result into an easy text report."""

    lines = [
        "=" * 60,
        "FedMSE Setup and Dataset Verification",
        "=" * 60,
        f"Python version: {summary['python_version']}",
        f"Dataset found: {summary['dataset_exists']}",
        f"Configuration found: {summary['configuration']['exists']}",
        f"Configuration valid: {summary['configuration']['valid_json']}",
        "",
        "Dependency Check",
        "-" * 60,
    ]

    for package_name, package_result in summary["dependencies"].items():
        if package_result["installed"]:
            lines.append(
                f"[OK] {package_name}: {package_result['version']}"
            )
        else:
            lines.append(f"[MISSING] {package_name}")

    lines.extend([
        "",
        "Client Dataset Check",
        "-" * 60,
    ])

    ready_client_count = 0

    for client in summary["clients"]:
        client_ready = (
            client["exists"]
            and all(
                folder["exists"] and folder["file_count"] > 0
                for folder in client["folders"].values()
            )
        )

        if client_ready:
            ready_client_count += 1

        status = "OK" if client_ready else "INCOMPLETE"
        lines.append(f"{client['client']}: {status}")

        for folder_name, folder_result in client["folders"].items():
            lines.append(
                f"  {folder_name}: "
                f"exists={folder_result['exists']}, "
                f"files={folder_result['file_count']}"
            )

    lines.extend([
        "",
        f"Ready clients: {ready_client_count}/10",
        f"Overall status: {summary['overall_status']}",
        "=" * 60,
    ])

    return "\n".join(lines)


def main():
    """Run all verification steps and save the results."""

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    dependencies = check_dependencies()
    configuration = check_configuration()
    clients = check_clients()

    dependencies_ready = all(
        result["installed"]
        for result in dependencies.values()
    )

    clients_ready = all(
        client["exists"]
        and all(
            folder["exists"] and folder["file_count"] > 0
            for folder in client["folders"].values()
        )
        for client in clients
    )

    dataset_exists = DATASET_PATH.exists()

    setup_ready = (
        dependencies_ready
        and configuration["exists"]
        and configuration["valid_json"]
        and dataset_exists
        and clients_ready
    )

    summary = {
        "python_version": sys.version.split()[0],
        "dataset_path": str(DATASET_PATH),
        "dataset_exists": dataset_exists,
        "configuration": configuration,
        "dependencies": dependencies,
        "clients": clients,
        "overall_status": (
            "SETUP READY"
            if setup_ready
            else "SETUP INCOMPLETE"
        ),
    }

    json_file = OUTPUT_PATH / "setup_summary.json"
    text_file = OUTPUT_PATH / "setup_report.txt"

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    report = create_text_report(summary)

    with text_file.open("w", encoding="utf-8") as file:
        file.write(report)

    print(report)
    print(f"\nJSON report saved: {json_file}")
    print(f"Text report saved: {text_file}")

    return 0 if setup_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())