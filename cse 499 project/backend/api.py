from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STATUS_FILE = DATA_DIR / "status.json"
ALERTS_FILE = DATA_DIR / "alerts.json"

DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PRIVATE SHIELD API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AlertIn(BaseModel):
    type: str
    confidence: float = 0.0
    device_id: str | None = None
    message: str | None = None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    import json
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def get_status() -> dict[str, Any]:
    default_status = {
        "running": True,
        "devices_online": 0,
        "messages_last_10s": 0,
        "alerts_fired": 0,
        "last_msg": None,
        "updated_at": now_iso(),
    }
    return read_json(STATUS_FILE, default_status)


@app.get("/alerts")
def get_alerts() -> list[dict[str, Any]]:
    alerts = read_json(ALERTS_FILE, [])
    return alerts[-20:]


@app.post("/alert")
def create_alert(alert: AlertIn) -> dict[str, Any]:
    alerts = read_json(ALERTS_FILE, [])

    new_alert = alert.model_dump()
    new_alert["timestamp"] = now_iso()

    alerts.append(new_alert)
    write_json(ALERTS_FILE, alerts[-100:])

    status = get_status()
    status["alerts_fired"] = int(status.get("alerts_fired", 0)) + 1
    status["updated_at"] = now_iso()
    write_json(STATUS_FILE, status)

    return {"saved": True, "alert": new_alert}


@app.post("/model/{model_name}")
def select_model(model_name: str) -> dict[str, str]:
    selected = {"selected_model": model_name, "updated_at": now_iso()}
    write_json(DATA_DIR / "selected_model.json", selected)
    return selected