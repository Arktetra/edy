from dotenv import load_dotenv
from pathlib import Path

import os

def load_env_variables():
    current_path = Path(__file__).resolve()
    root = current_path.parent.parent

    dotenv_path = root / ".env"

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment variables from: {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Ensure that it is in the project root")