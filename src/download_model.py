"""
download_model.py - Resumable and Fault-Tolerant Base LLM Downloader.

Features:
  * Cleans up stale Hugging Face lock files automatically.
  * Uses snapshot_download with resume_download=True for reliable downloads.
  * Shows real-time progress and verifies shard completeness.
  * Can be stopped and resumed at any time without losing downloaded bytes.
  * Enables 100% offline local training once completed.
"""

import argparse
import glob
import os
import sys
import time


def clear_stale_locks(model_id: str):
    """Remove any orphaned lock files from previous killed downloads."""
    safe_name = "models--" + model_id.replace("/", "--")
    lock_dir = os.path.expanduser(f"~/.cache/huggingface/hub/.locks/{safe_name}")
    if os.path.exists(lock_dir):
        locks = glob.glob(os.path.join(lock_dir, "*.lock"))
        for lock in locks:
            try:
                os.remove(lock)
                print(f"Cleared stale lock: {os.path.basename(lock)}")
            except Exception:
                pass


def is_model_complete(model_id: str) -> bool:
    """Check if all weight shards are already cached and non-empty."""
    safe_name = "models--" + model_id.replace("/", "--")
    snap_dir = os.path.expanduser(f"~/.cache/huggingface/hub/{safe_name}/snapshots")
    if not os.path.exists(snap_dir):
        return False

    snapshots = os.listdir(snap_dir)
    if not snapshots:
        return False

    for snap in snapshots:
        folder = os.path.join(snap_dir, snap)
        index_file = os.path.join(folder, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shards = set(data.get("weight_map", {}).values())
                all_present = True
                for shard in shards:
                    shard_path = os.path.join(folder, shard)
                    if not os.path.exists(shard_path) or os.path.getsize(shard_path) < 1024 * 1024:
                        all_present = False
                        break
                if all_present:
                    return True
            except Exception:
                pass

        # Also check single model.safetensors or bin
        single_safe = os.path.join(folder, "model.safetensors")
        if os.path.exists(single_safe) and os.path.getsize(single_safe) > 100 * 1024 * 1024:
            return True

    return False


def download_base_model(model_id: str = "Qwen/Qwen2.5-3B-Instruct", max_retries: int = 5):
    """Download base model with automatic retries and resume capability."""
    print(f"\n{'='*70}")
    print(f" VERIFYING / DOWNLOADING BASE LLM: {model_id}")
    print(f"{'='*70}")

    clear_stale_locks(model_id)

    if is_model_complete(model_id):
        print(f"\n[OK] Model '{model_id}' is already fully downloaded and verified!")
        print("You can run training completely OFFLINE without any internet connection.\n")
        return True

    print("Model weights not yet complete. Starting resumable download...")
    print("NOTE: If your internet disconnects, simply run this script again.")
    print("It will automatically resume from the exact byte where it paused.\n")

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from huggingface_hub import snapshot_download

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Download attempt {attempt}/{max_retries}...")
            cached_path = snapshot_download(
                repo_id=model_id,
                resume_download=True,
                max_workers=2,
            )
            print(f"\n[SUCCESS] Successfully downloaded to:\n  {cached_path}")
            if is_model_complete(model_id):
                print("\n[VERIFIED] All weight shards are present and verified.")
                print("Your local machine is now ready for offline training!")
                return True
        except Exception as e:
            print(f"\n[WARNING] Attempt {attempt} interrupted: {e}")
            clear_stale_locks(model_id)
            if attempt < max_retries:
                wait_sec = 5
                print(f"Retrying in {wait_sec} seconds (resuming download)...")
                time.sleep(wait_sec)
            else:
                print("\n[ERROR] Download could not complete after maximum retries.")
                print("Please check your internet connection and re-run:")
                print(f"  python src/download_model.py --model_id {model_id}")
                return False

    return is_model_complete(model_id)


def main():
    parser = argparse.ArgumentParser(description="Resumable Model Downloader")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--check_only", action="store_true", help="Only check if already downloaded")
    args = parser.parse_args()

    if args.check_only:
        complete = is_model_complete(args.model_id)
        if complete:
            print(f"[OK] {args.model_id} is completely downloaded.")
            sys.exit(0)
        else:
            print(f"[INCOMPLETE] {args.model_id} is not fully downloaded.")
            sys.exit(1)

    success = download_base_model(args.model_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
