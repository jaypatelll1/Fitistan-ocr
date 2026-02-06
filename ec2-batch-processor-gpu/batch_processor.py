#!/usr/bin/env python3
"""
EC2 Batch Processor - OCR & Face Embeddings (GPU Version)
Process S3 images and save results to PostgreSQL.

Usage:
    python3 batch_processor.py s3://bucket/prefix/ [--max N] [--dry-run]
"""

import os
import sys
import json
import argparse
import tempfile
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

import cv2
import boto3
import numpy as np
import psycopg
from PIL import Image
from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis
from pgvector.psycopg import register_vector

# Config
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE = 1200
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")


def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def error(msg: str):
    print(f"[ERROR] {msg}", flush=True)


class Processor:
    """OCR and Face embedding extraction."""

    def __init__(self, use_gpu: bool = True):
        self._ocr = None
        self._face = None
        self.use_gpu = use_gpu

    @property
    def ocr(self):
        if self._ocr is None:
            log("Loading OCR model...")
            self._ocr = PaddleOCR(
                lang="ch",
                ocr_version="PP-OCRv4",
                use_textline_orientation=True,
                device="gpu" if self.use_gpu else "cpu",
                show_log=False,
            )
        return self._ocr

    @property
    def face(self):
        if self._face is None:
            log("Loading face model...")
            self._face = FaceAnalysis(name="buffalo_l")
            ctx = 0 if self.use_gpu else -1
            self._face.prepare(ctx_id=ctx, det_size=(1024, 1024), det_thresh=0.4)
        return self._face

    def extract_bibs(self, path: str) -> List[str]:
        result = self.ocr.ocr(path, cls=True)
        bibs = []
        if result and result[0]:
            for det in result[0]:
                text = det[1][0].strip()
                if text.isdigit():
                    bibs.append(text)
        return bibs

    def extract_faces(self, path: str) -> List[Dict]:
        img = cv2.imread(path)
        if img is None:
            return []
        faces = self.face.get(img)
        return [
            {"index": i + 1, "embedding": f.embedding.astype(np.float32).tolist()}
            for i, f in enumerate(faces)
        ]


class S3:
    """S3 operations."""

    def __init__(self):
        self.client = boto3.client("s3", region_name=AWS_REGION)

    def list_images(self, bucket: str, prefix: str, max_count: Optional[int] = None) -> List[str]:
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if os.path.splitext(key)[1].lower() in IMAGE_EXTENSIONS:
                    keys.append(key)
                    if max_count and len(keys) >= max_count:
                        return keys
        return keys

    def download(self, bucket: str, key: str, path: str):
        self.client.download_file(bucket, key, path)


class DB:
    """PostgreSQL operations."""

    def __init__(self):
        self.conn = psycopg.connect(
            host=os.environ["DB_HOST"],
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            sslmode=os.getenv("DB_SSLMODE", "require"),
        )
        register_vector(self.conn)

    def save_bibs(self, image_name: str, bibs: List[str], url: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_photo_bib_number (image_name, bib_number, url, event_code)
                VALUES (%s, %s, %s, NULL)
                ON CONFLICT (image_name) DO UPDATE SET
                    bib_number = EXCLUDED.bib_number, url = EXCLUDED.url, last_modified = now()
                """,
                (image_name, json.dumps(bibs), url),
            )
        self.conn.commit()

    def save_face(self, image_name: str, idx: int, embedding: List[float], url: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_face_embedding (image_name, face_index, embedding, event_code, url)
                VALUES (%s, %s, %s, NULL, %s)
                ON CONFLICT (image_name, face_index) DO UPDATE SET
                    embedding = EXCLUDED.embedding, url = EXCLUDED.url, last_modified = now()
                """,
                (image_name, idx, embedding, url),
            )
        self.conn.commit()

    def close(self):
        self.conn.close()


def resize_image(src: str, dst: str):
    with Image.open(src) as img:
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        img.save(dst)


def parse_s3_path(path: str) -> tuple:
    path = path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def process_image(processor: Processor, s3: S3, db: Optional[DB], bucket: str, key: str) -> Dict:
    result = {"key": key, "bibs": [], "faces": 0, "error": None}
    ext = os.path.splitext(key)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
    resized_path = tmp_path.replace(ext, f"_resized{ext}")

    try:
        s3.download(bucket, key, tmp_path)
        resize_image(tmp_path, resized_path)

        bibs = processor.extract_bibs(resized_path)
        faces = processor.extract_faces(resized_path)

        result["bibs"] = bibs
        result["faces"] = len(faces)

        if db:
            url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{key}"
            filename = os.path.basename(key)
            db.save_bibs(key, bibs, url)
            for face in faces:
                db.save_face(filename, face["index"], face["embedding"], url)

    except Exception as e:
        result["error"] = str(e)
    finally:
        for p in [tmp_path, resized_path]:
            if os.path.exists(p):
                os.remove(p)

    return result


def main():
    parser = argparse.ArgumentParser(description="Process S3 images for OCR and face embeddings")
    parser.add_argument("s3_path", nargs="?", help="S3 path (s3://bucket/prefix/) - or set S3_PATH env")
    parser.add_argument("--max", type=int, help="Max images to process")
    parser.add_argument("--dry-run", action="store_true", help="Skip database writes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    # Get S3 path from arg or env
    s3_path = args.s3_path or os.getenv("S3_PATH")
    if not s3_path:
        error("S3 path required. Provide as argument or set S3_PATH env")
        sys.exit(1)

    bucket, prefix = parse_s3_path(s3_path)
    log(f"Processing: s3://{bucket}/{prefix}")

    s3 = S3()
    processor = Processor(use_gpu=not args.cpu)
    db = None

    if not args.dry_run:
        log("Connecting to database...")
        db = DB()

    try:
        keys = s3.list_images(bucket, prefix, args.max)
        total = len(keys)
        log(f"Found {total} images")

        if not keys:
            error("No images found")
            return

        success, failed = 0, 0
        for i, key in enumerate(keys, 1):
            result = process_image(processor, s3, db, bucket, key)

            if result["error"]:
                failed += 1
                error(f"[{i}/{total}] ✗ {key}: {result['error']}")
            else:
                success += 1
                log(f"[{i}/{total}] ✓ {key} | bibs={result['bibs']} faces={result['faces']}")

        log(f"Complete: {success} success, {failed} failed")

    finally:
        if db:
            db.close()


if __name__ == "__main__":
    main()
