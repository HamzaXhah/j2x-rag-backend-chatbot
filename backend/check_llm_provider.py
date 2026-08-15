"""Check the configured Alibaba Model Studio LLM without starting the application.

By default, this script reads the environment file beside it at ``backend/.env``.
An alternative file can be supplied with ``--env-file``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a minimal request directly to the configured LLM provider."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f"Environment file to load (default: {DEFAULT_ENV_FILE})",
    )
    return parser.parse_args()


def required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        print(f"CONFIGURATION ERROR: {name} is missing or empty.")
        raise SystemExit(2)
    return value


def main() -> int:
    args = parse_args()
    env_file = args.env_file.expanduser().resolve()

    if not env_file.is_file():
        print(f"CONFIGURATION ERROR: Environment file not found: {env_file}")
        return 2

    # The selected file must win over variables inherited from the current shell.
    load_dotenv(env_file, override=True)

    api_key = required_setting("DASHSCOPE_API_KEY")
    base_url = required_setting("DASHSCOPE_BASE_URL").rstrip("/")
    model = os.getenv("QWEN_MODEL", "qwen3.7-plus").strip() or "qwen3.7-plus"
    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]

    print(f"Environment file : {env_file}")
    print(f"Base URL         : {base_url}")
    print(f"Model            : {model}")
    print(f"API key          : set (length={len(api_key)}, sha256={key_fingerprint}...)")
    print("Sending a minimal OpenAI-compatible Chat Completions request...")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=16,
            extra_body={"enable_thinking": False},
        )
        answer = response.choices[0].message.content or ""
        print(f"SUCCESS: Provider responded: {answer.strip()!r}")
        return 0

    except openai.AuthenticationError as exc:
        print(f"AUTHENTICATION FAILED: {exc}")
        print("Check that the API key belongs to the endpoint's region/workspace.")
        return 3
    except openai.PermissionDeniedError as exc:
        print(f"ACCESS DENIED: {exc}")
        print(f"The key may not have permission to use model {model!r}.")
        return 4
    except openai.NotFoundError as exc:
        print(f"MODEL OR ENDPOINT NOT FOUND: {exc}")
        print("Check DASHSCOPE_BASE_URL and QWEN_MODEL.")
        return 5
    except openai.APIConnectionError as exc:
        print(f"CONNECTION FAILED: {exc}")
        print("Check the endpoint, DNS, proxy, firewall, and TLS configuration.")
        return 6
    except openai.APIStatusError as exc:
        print(f"PROVIDER ERROR: HTTP {exc.status_code}: {exc}")
        return 7
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        return 8


if __name__ == "__main__":
    sys.exit(main())
