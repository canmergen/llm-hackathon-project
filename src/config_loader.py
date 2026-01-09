import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path("config.yaml")

def load_config() -> Dict[str, Any]:
    """Load configuration from yaml file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Singleton inst
try:
    config = load_config()
except Exception as e:
    print(f"⚠️ Could not load config: {e}")
    config = {}
