import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from dotenv import load_dotenv


def deploy(space_id: str, token: str, private: bool) -> None:
    root = Path(__file__).resolve().parent
    api = HfApi(token=token)

    # Create (or reuse) a Docker Space, then upload repository contents.
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        private=private,
        space_sdk="docker",
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(root),
        ignore_patterns=[
            "venv/*",
            "__pycache__/*",
            ".cache/*",
            ".git/*",
            "*.sqlite",
            "*.pyc",
            "*.pyo",
            ".env",
            "ci_policy.json",
            "full_policy.json",
            "rl_policy.json",
        ],
    )

    print(f"Deployment upload complete: https://huggingface.co/spaces/{space_id}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Deploy this project to a Hugging Face Docker Space.")
    parser.add_argument("--space-id", required=True, help="Space ID in form 'username/space-name'")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument("--private", action="store_true", help="Create/update the Space as private")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Missing token. Set HF_TOKEN or pass --token.")

    deploy(space_id=args.space_id, token=args.token, private=args.private)


if __name__ == "__main__":
    main()
