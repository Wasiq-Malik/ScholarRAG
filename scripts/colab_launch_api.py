from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_RUN_DIR = Path(
    "/content/drive/MyDrive/scholarrag/"
    "open_arxiv_embeddinggemma_fact_check_cs_100k_from_2020_recent_ivfflat"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the ScholarRAG FastAPI app in Colab using FAISS/SQLite "
            "artifacts saved in Google Drive."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--faiss-index", type=Path, default=None)
    parser.add_argument("--sqlite", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemma-4-31b-it"))
    parser.add_argument(
        "--tunnel",
        choices=["none", "ngrok"],
        default="ngrok",
        help="Expose the local Colab server. ngrok requires NGROK_AUTHTOKEN.",
    )
    parser.add_argument(
        "--install-ngrok",
        action="store_true",
        help="Install pyngrok if it is missing.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print the resolved ScholarRAG environment before starting.",
    )
    return parser.parse_args()


def resolve_artifacts(args: argparse.Namespace) -> tuple[Path, Path]:
    faiss_index = args.faiss_index or args.run_dir / "faiss" / "open_arxiv_papers.faiss"
    sqlite_path = args.sqlite or args.run_dir / "metadata" / "open_arxiv_papers.sqlite"
    missing = [str(path) for path in [faiss_index, sqlite_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nMount Google Drive first and confirm --run-dir points at the completed FAISS run."
        )
    return faiss_index, sqlite_path


def configure_environment(args: argparse.Namespace, faiss_index: Path, sqlite_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SCHOLARRAG_VECTOR_BACKEND": "faiss",
            "SCHOLARRAG_FAISS_INDEX_PATH": str(faiss_index),
            "SCHOLARRAG_FAISS_SQLITE_PATH": str(sqlite_path),
            "SCHOLARRAG_FAISS_NPROBE": str(args.nprobe),
            "SCHOLARRAG_RETRIEVAL_CONTEXT_K": str(args.top_k),
            "SCHOLARRAG_RETRIEVAL_CANDIDATE_K": str(args.candidate_k),
            "SCHOLARRAG_GEMINI_MODEL": args.gemini_model,
        }
    )
    if env.get("GEMINI_API_KEY") and not env.get("SCHOLARRAG_GEMINI_API_KEY"):
        env["SCHOLARRAG_GEMINI_API_KEY"] = env["GEMINI_API_KEY"]
    if env.get("HF_TOKEN") and not env.get("SCHOLARRAG_HF_TOKEN"):
        env["SCHOLARRAG_HF_TOKEN"] = env["HF_TOKEN"]
    return env


def maybe_install_pyngrok() -> None:
    try:
        import pyngrok  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])


def start_ngrok(port: int, *, install_ngrok: bool) -> str | None:
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        print("NGROK_AUTHTOKEN is not set; starting local server without a public tunnel.")
        return None
    if install_ngrok:
        maybe_install_pyngrok()
    try:
        from pyngrok import ngrok
    except ImportError:
        print("pyngrok is not installed. Re-run with --install-ngrok or install it in Colab.")
        return None
    ngrok.set_auth_token(token)
    tunnel = ngrok.connect(port, bind_tls=True)
    public_url = str(tunnel.public_url)
    print(f"ngrok public URL: {public_url}")
    print(f"Health check: {public_url}/health")
    return public_url


def wait_for_health(port: int, timeout_seconds: int = 120) -> bool:
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except URLError:
            time.sleep(1)
    return False


def main() -> int:
    args = parse_args()
    faiss_index, sqlite_path = resolve_artifacts(args)
    env = configure_environment(args, faiss_index, sqlite_path)

    if args.print_env:
        keys = [
            "SCHOLARRAG_VECTOR_BACKEND",
            "SCHOLARRAG_FAISS_INDEX_PATH",
            "SCHOLARRAG_FAISS_SQLITE_PATH",
            "SCHOLARRAG_FAISS_NPROBE",
            "SCHOLARRAG_RETRIEVAL_CONTEXT_K",
            "SCHOLARRAG_RETRIEVAL_CANDIDATE_K",
            "SCHOLARRAG_GEMINI_MODEL",
        ]
        for key in keys:
            print(f"{key}={env.get(key)}")

    public_url = None
    if args.tunnel == "ngrok":
        public_url = start_ngrok(args.port, install_ngrok=args.install_ngrok)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "scholarrag.api.app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print("Starting ScholarRAG API...")
    print("Local URL: http://127.0.0.1:%d" % args.port)
    if public_url:
        print(f"Public URL: {public_url}")
    print("Press Ctrl-C to stop.")

    process = subprocess.Popen(command, env=env, cwd=REPO_ROOT)
    try:
        if wait_for_health(args.port):
            print("ScholarRAG API health check passed.")
        else:
            print("Warning: local health check did not pass before timeout.")
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
