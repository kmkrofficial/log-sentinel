import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcore.setup_manager import SetupError, provision_assets


def _progress(payload: dict) -> None:
    message = payload.get("log")
    print(message if message else payload.get("status", "Provisioning model assets."))

def main():
    print("--- LogSentinel Model Downloader ---")
    try:
        result = provision_assets(model_keys=["encoder", "llama"], callback=_progress)
    except SetupError as error:
        print(f"Model provisioning failed: {error}")
        return 1
    print(f"--- Model download check complete: {result} ---")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())