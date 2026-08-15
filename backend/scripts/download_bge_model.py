"""Download BAAI BGE Large into the backend's local model directory."""

import os
import tempfile
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
configured_path = Path(os.getenv("BGE_MODEL_PATH", "./models/bge-large-en-v1.5"))
MODEL_PATH = configured_path if configured_path.is_absolute() else BACKEND_DIR / configured_path


def main() -> None:
    safetensors_path = MODEL_PATH / "model.safetensors"
    pytorch_path = MODEL_PATH / "pytorch_model.bin"
    completion_marker = MODEL_PATH / ".download-complete"
    has_weights = (
        safetensors_path.is_file()
        and safetensors_path.stat().st_size > 1_000_000_000
    ) or (
        pytorch_path.is_file()
        and pytorch_path.stat().st_size > 1_000_000_000
    )
    if (
        completion_marker.is_file()
        and (MODEL_PATH / "config.json").is_file()
        and has_weights
    ):
        print(f"BGE model is already available at {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_NAME} to {MODEL_PATH}...")
    from sentence_transformers import SentenceTransformer

    with tempfile.TemporaryDirectory(
        prefix=".bge-download-",
        dir=MODEL_PATH.parent,
    ) as cache_dir:
        model = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir)
        model.save_pretrained(str(MODEL_PATH))
    completion_marker.write_text(f"{MODEL_NAME}\n", encoding="utf-8")
    print(f"BGE model downloaded successfully to {MODEL_PATH}")


if __name__ == "__main__":
    main()
