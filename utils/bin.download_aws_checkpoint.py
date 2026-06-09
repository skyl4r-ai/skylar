#!/usr/bin/env python3
"""
Download a model checkpoint from S3.

Usage:
  # Download best checkpoint
  python download_model.py \
    --checkpoint best \
    --s3_bucket mio-bucket \
    --s3_prefix <your-prefix> \
    --s3_region eu-west-1 \
    --output ./my_model

  # Download a specific step
  python download_model.py \
    --checkpoint step_5000 \
    --s3_bucket mio-bucket \
    --s3_prefix <your-prefix> \
    --s3_region eu-west-1 \
    --output ./my_model

  # Download final checkpoint
  python download_model.py \
    --checkpoint best \
    --s3_bucket <your-bucket> \
    --s3_prefix <your-prefix> \
    --s3_region eu-south-1
    --output ./checkpoints/Skylar-100M-Base

  # List available checkpoints
  python download_model.py \
    --list \
    --s3_bucket mio-bucket \
    --s3_prefix <your-prefix> \
    --s3_region eu-west-1

Env vars required:
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

MAX_RETRIES: int = 5
RETRY_BASE_SEC: float = 2.0


def get_s3_client(bucket: str, region: str):
    """Create boto3 S3 client."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("  ✗ boto3 required: pip install boto3")
        sys.exit(1)

    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        print("  ✗ Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars")
        sys.exit(1)

    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(max_pool_connections=10),
    )


def list_checkpoints(client, bucket: str, prefix: str) -> list[str]:
    """List available checkpoint names under prefix/checkpoints/."""
    ckpt_prefix = f"{prefix}/checkpoints/" if prefix else "checkpoints/"
    paginator = client.get_paginator("list_objects_v2")

    names: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=ckpt_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            # e.g. "<your-prefix>/checkpoints/best/"
            name = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            names.add(name)

    return sorted(names)


def list_files_in_checkpoint(client, bucket: str, prefix: str, checkpoint: str) -> list[dict]:
    """List all files inside a checkpoint directory."""
    ckpt_prefix = f"{prefix}/checkpoints/{checkpoint}/" if prefix else f"checkpoints/{checkpoint}/"
    paginator = client.get_paginator("list_objects_v2")

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=ckpt_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(ckpt_prefix):]
            if filename:
                files.append({
                    "key": key,
                    "filename": filename,
                    "size": obj["Size"],
                })
    return files


def download_file(client, bucket: str, key: str, local_path: Path) -> None:
    """Download a single file with retry."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.download_file(bucket, key, str(local_path))
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BASE_SEC * (2 ** (attempt - 1))
            print(f"    ⚠ Retry {attempt}/{MAX_RETRIES} for {local_path.name} in {wait:.0f}s...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Download model checkpoint from S3")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint name: best, final, step_5000, etc.")
    parser.add_argument("--output", type=str, default=None,
                        help="Local output directory (default: ./checkpoint_name)")
    parser.add_argument("--list", action="store_true",
                        help="List available checkpoints and exit")

    parser.add_argument("--s3_bucket", required=True, help="AWS S3 bucket name")
    parser.add_argument("--s3_prefix", default="", help="Key prefix inside bucket")
    parser.add_argument("--s3_region", default="us-east-1", help="AWS region")

    args = parser.parse_args()

    client = get_s3_client(args.s3_bucket, args.s3_region)
    prefix = args.s3_prefix.strip("/") if args.s3_prefix else ""

    # ── List mode ──
    if args.list:
        print(f"\n  Checkpoints in s3://{args.s3_bucket}/{prefix}/checkpoints/:\n")
        names = list_checkpoints(client, args.s3_bucket, prefix)
        if not names:
            print("    (none found)")
        else:
            for name in names:
                files = list_files_in_checkpoint(client, args.s3_bucket, prefix, name)
                total_mb = sum(f["size"] for f in files) / 1e6
                print(f"    {name:20s}  {len(files)} files  {total_mb:8.1f} MB")
        print()
        return

    # ── Download mode ──
    if not args.checkpoint:
        parser.error("--checkpoint is required (or use --list to see available)")

    out_dir = Path(args.output) if args.output else Path(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading checkpoint '{args.checkpoint}' from s3://{args.s3_bucket}/{prefix}/")
    print(f"  Output: {out_dir}\n")

    files = list_files_in_checkpoint(client, args.s3_bucket, prefix, args.checkpoint)
    if not files:
        print(f"  ✗ No files found for checkpoint '{args.checkpoint}'")
        print(f"    Use --list to see available checkpoints")
        sys.exit(1)

    total_bytes = sum(f["size"] for f in files)
    print(f"  {len(files)} files, {total_bytes / 1e6:.1f} MB total\n")

    t0 = time.time()
    for i, f in enumerate(files, 1):
        local_path = out_dir / f["filename"]
        size_mb = f["size"] / 1e6
        print(f"  [{i}/{len(files)}] {f['filename']} ({size_mb:.1f} MB)...", end=" ", flush=True)
        download_file(client, args.s3_bucket, f["key"], local_path)
        print("✓")

    elapsed = time.time() - t0
    print(f"\n  ✅ Downloaded to {out_dir} in {elapsed:.1f}s")
    print(f"     Use with: python train.py --resume {out_dir}\n")


if __name__ == "__main__":
    main()