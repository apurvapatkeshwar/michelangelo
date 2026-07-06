"""Flat-upload a CustomTritonPackager output directory to MinIO for KServe.

KServe's S3 storage-initializer downloads `storageUri` as a directory
prefix — it does not untar anything. The standard pipeline path
(MinioStorageBackend.upload(), used by examples/bert_cola/serve.py) tars the
whole deployable dir into a single `__dir__.tar` object, which works for the
existing Triton backend (a plain container that reads a local mount) but is
NOT a valid layout for KServe's predictor. This script uploads the same
directory tree flat, file by file, under a `kserve-repo/` prefix instead.

This is a standalone demo/dev tool, not wired into the pipeline — see the
"Known limitations" section of ../README.md for why it hasn't been folded
into MinioStorageBackend / serve.py yet.

Usage:
    python upload_model_repo.py \
        --local-dir /tmp/bert_cola_deployable_xxxx \
        --bucket deploy-models \
        --prefix bert-cola/kserve-repo \
        --endpoint host.docker.internal:9091
"""

from __future__ import annotations

import argparse
import os

import boto3


def upload_directory(local_dir: str, bucket: str, prefix: str, endpoint: str, access_key: str, secret_key: str) -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{endpoint}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    for root, _dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            key = f"{prefix.rstrip('/')}/{rel_path}"
            print(f"uploading {local_path} -> s3://{bucket}/{key}")
            s3.upload_file(local_path, bucket, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True, help="CustomTritonPackager output dir")
    parser.add_argument("--bucket", default="deploy-models")
    parser.add_argument("--prefix", default="bert-cola/kserve-repo")
    parser.add_argument("--endpoint", default="host.docker.internal:9091")
    parser.add_argument("--access-key", default=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"))
    parser.add_argument("--secret-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    args = parser.parse_args()

    upload_directory(args.local_dir, args.bucket, args.prefix, args.endpoint, args.access_key, args.secret_key)


if __name__ == "__main__":
    main()
