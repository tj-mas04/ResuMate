import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load environment variables on import
load_dotenv()

# Project paths
REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "resumate.db"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_groq_client() -> Groq | None:
    """Return a Groq client if API key is configured, else None."""
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def get_languagetool_cache_dir() -> Path:
    """Resolve writable cache directory for LanguageTool models."""
    env_dir = os.getenv("LANGUAGE_TOOL_CACHE_DIR")
    if env_dir:
        return Path(env_dir)

    # Prefer repo-local LanguageTool folder
    local = REPO_ROOT / "LanguageTool"
    try:
        local.mkdir(parents=True, exist_ok=True)
        return local
    except PermissionError:
        # Fallback to user home if repo not writable
        home = Path.home() / ".resumate" / "languagetool"
        home.mkdir(parents=True, exist_ok=True)
        return home
