from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    dataset_dir: Path = data_dir / "dataset"

    # SEC requires a descriptive User-Agent with contact information
    # Replace the email below with your own email address

    user_agent: str = "BANA275-NLP-StudentProject/1.0 (contact: your@email)"
   
    # Conservative rate limit: 0.2s ≈ 5 requests/sec
    sleep_seconds: float = 0.2

settings = Settings()