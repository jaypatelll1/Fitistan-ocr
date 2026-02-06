#!/usr/bin/env python3
"""
EC2 Batch Processor - OCR & Face Embeddings (GPU Only)
Usage: python3 batch_processor.py s3://bucket/prefix/ [--max N] [--workers N] [--dry-run]
"""

import os
import sys
import json
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor

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
DEFAULT_WORKERS = 4


def log(msg):
    print(f"[INFO] {msg}", flush=True)


def error(msg):
    print(f"[ERROR] {msg}", flush=True)


def init_models():
    """Initialize OCR and Face models on GPU."""
    log("Loading OCR model (GPU)...")
    ocr = PaddleOCR(
        lang="ch",
        ocr_version="PP-OCRv4",
        rec_model_dir="ch_PP-OCRv4_server_rec",
        device="gpu",
        use_angle_cls=False,
        det_db_thresh=0.1,
        det_db_box_thresh=0.1,
        drop_score=0.1,
        det_limit_side_len=1280,
        show_log=False,
    )

    log("Loading face model (GPU)...")
    face = FaceAnalysis(name="buffalo_l")
    face.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.4)

    return ocr, face


def extract_bibs(ocr, path):
    """Extract numeric bib numbers from image."""
    result = ocr.ocr(path, cls=False)
    bibs = []
    if result and result[0]:
        for det in result[0]:
            text = det[1][0].strip()
            if text.isdigit():
                bibs.append(text)
    return bibs


def extract_faces(face, path):
    """Extract face embeddings from image."""
    img = cv2.imread(path)
    if img is None:
        return []
    faces = face.get(img)
    return [{"index": i + 1, "embedding": f.embedding.astype(np.float32).tolist()} for i, f in enumerate(faces)]


def list_s3_images(s3, bucket, prefix, max_count=None):
    """List image files from S3."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if os.path.splitext(key)[1].lower() in IMAGE_EXTENSIONS:
                keys.append(key)
                if max_count and len(keys) >= max_count:
                    return keys
    return keys


def connect_db():
    """Connect to PostgreSQL."""
    conn = psycopg.connect(
        host=os.environ["DB_HOST"],
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        sslmode=os.getenv("DB_SSLMODE", "require"),
    )
    register_vector(conn)
    return conn


def save_to_db(conn, key, bibs, faces, bucket):
    """Save bibs and faces to database."""
    url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{key}"
    filename = os.path.basename(key)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO event_photo_bib_number (image_name, bib_number, url, event_code)
            VALUES (%s, %s, %s, NULL)
            ON CONFLICT (image_name) DO UPDATE SET
                bib_number = EXCLUDED.bib_number, url = EXCLUDED.url, last_modified = now()
        """, (key, json.dumps(bibs), url))

        for face in faces:
            cur.execute("""
                INSERT INTO event_face_embedding (image_name, face_index, embedding, event_code, url)
                VALUES (%s, %s, %s, NULL, %s)
                ON CONFLICT (image_name, face_index) DO UPDATE SET
                    embedding = EXCLUDED.embedding, url = EXCLUDED.url, last_modified = now()
            """, (filename, face["index"], face["embedding"], url))

    conn.commit()


def resize_image(src, dst):
    """Resize image to max dimension."""
    with Image.open(src) as img:
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        img.save(dst)


def download_and_resize(s3, bucket, key):
    """Download from S3 and resize image. Returns (key, resized_path) or (key, None, error)."""
    ext = os.path.splitext(key)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
    resized_path = tmp_path.replace(ext, f"_resized{ext}")

    try:
        s3.download_file(bucket, key, tmp_path)
        resize_image(tmp_path, resized_path)
        os.remove(tmp_path)
        return (key, resized_path, None)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return (key, None, str(e))


def process_batch(ocr, face, s3, conn, bucket, keys, workers):
    """Process a batch of images: parallel download, sequential GPU inference."""
    results = []

    # Parallel download and resize
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_and_resize, s3, bucket, key) for key in keys]
        downloaded = [f.result() for f in futures]

    # Sequential GPU processing
    for key, resized_path, download_error in downloaded:
        if download_error:
            results.append({"key": key, "bibs": [], "faces": 0, "error": download_error})
            continue

        try:
            bibs = extract_bibs(ocr, resized_path)
            faces = extract_faces(face, resized_path)

            if conn:
                save_to_db(conn, key, bibs, faces, bucket)

            results.append({"key": key, "bibs": bibs, "faces": len(faces), "error": None})
        except Exception as e:
            results.append({"key": key, "bibs": [], "faces": 0, "error": str(e)})
        finally:
            if resized_path and os.path.exists(resized_path):
                os.remove(resized_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Process S3 images for OCR and face embeddings (GPU)")
    parser.add_argument("s3_path", nargs="?", help="S3 path (s3://bucket/prefix/) or set S3_PATH env")
    parser.add_argument("--max", type=int, help="Max images to process")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel downloads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true", help="Skip database writes")
    args = parser.parse_args()

    # Get S3 path
    s3_path = args.s3_path or os.getenv("S3_PATH")
    if not s3_path:
        error("S3 path required. Provide as argument or set S3_PATH env")
        sys.exit(1)

    # Parse S3 path
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    bucket, prefix = parts[0], parts[1] if len(parts) > 1 else ""

    log(f"Processing: s3://{bucket}/{prefix}")
    log(f"Workers: {args.workers}")

    # Initialize
    s3 = boto3.client("s3", region_name=AWS_REGION)
    ocr, face = init_models()
    conn = None if args.dry_run else connect_db()

    try:
        keys = list_s3_images(s3, bucket, prefix, args.max)
        total = len(keys)
        log(f"Found {total} images")

        if not keys:
            error("No images found")
            return

        success, failed = 0, 0

        # Process in batches
        for i in range(0, total, args.workers):
            batch_keys = keys[i:i + args.workers]
            results = process_batch(ocr, face, s3, conn, bucket, batch_keys, args.workers)

            for j, result in enumerate(results):
                idx = i + j + 1
                if result["error"]:
                    failed += 1
                    error(f"[{idx}/{total}] ✗ {result['key']}: {result['error']}")
                else:
                    success += 1
                    log(f"[{idx}/{total}] ✓ {result['key']} | bibs={result['bibs']} faces={result['faces']}")

        log(f"Complete: {success} success, {failed} failed")

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
