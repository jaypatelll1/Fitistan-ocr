#!/usr/bin/env python3
"""
EC2 Batch Processor - OCR & Face Embeddings
Usage: python3 batch_processor.py s3://bucket/prefix/ [--max N] [--dry-run] [--cpu]
"""

import os
import sys
import json
import argparse
import tempfile

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


def log(msg):
    print(f"[INFO] {msg}", flush=True)


def error(msg):
    print(f"[ERROR] {msg}", flush=True)


def init_models(use_gpu=True):
    """Initialize OCR and Face models."""
    log("Loading OCR model...")
    ocr = PaddleOCR(
        lang="ch",
        ocr_version="PP-OCRv4",
        use_textline_orientation=True,
        device="gpu" if use_gpu else "cpu",
        show_log=False,
    )

    log("Loading face model...")
    face = FaceAnalysis(name="buffalo_l")
    face.prepare(ctx_id=0 if use_gpu else -1, det_size=(1024, 1024), det_thresh=0.4)

    return ocr, face


def extract_bibs(ocr, path):
    """Extract numeric bib numbers from image."""
    result = ocr.ocr(path, cls=True)
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
        # Save bibs
        cur.execute("""
            INSERT INTO event_photo_bib_number (image_name, bib_number, url, event_code)
            VALUES (%s, %s, %s, NULL)
            ON CONFLICT (image_name) DO UPDATE SET
                bib_number = EXCLUDED.bib_number, url = EXCLUDED.url, last_modified = now()
        """, (key, json.dumps(bibs), url))

        # Save faces
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


def process_image(ocr, face, s3, conn, bucket, key):
    """Process single image: download, OCR, face detection, save."""
    ext = os.path.splitext(key)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
    resized_path = tmp_path.replace(ext, f"_resized{ext}")

    try:
        s3.download_file(bucket, key, tmp_path)
        resize_image(tmp_path, resized_path)

        bibs = extract_bibs(ocr, resized_path)
        faces = extract_faces(face, resized_path)

        if conn:
            save_to_db(conn, key, bibs, faces, bucket)

        return {"bibs": bibs, "faces": len(faces), "error": None}

    except Exception as e:
        return {"bibs": [], "faces": 0, "error": str(e)}

    finally:
        for p in [tmp_path, resized_path]:
            if os.path.exists(p):
                os.remove(p)


def main():
    parser = argparse.ArgumentParser(description="Process S3 images for OCR and face embeddings")
    parser.add_argument("s3_path", nargs="?", help="S3 path (s3://bucket/prefix/) or set S3_PATH env")
    parser.add_argument("--max", type=int, help="Max images to process")
    parser.add_argument("--dry-run", action="store_true", help="Skip database writes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
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

    # Initialize
    s3 = boto3.client("s3", region_name=AWS_REGION)
    ocr, face = init_models(use_gpu=not args.cpu)
    conn = None if args.dry_run else connect_db()

    try:
        keys = list_s3_images(s3, bucket, prefix, args.max)
        total = len(keys)
        log(f"Found {total} images")

        if not keys:
            error("No images found")
            return

        success, failed = 0, 0
        for i, key in enumerate(keys, 1):
            result = process_image(ocr, face, s3, conn, bucket, key)

            if result["error"]:
                failed += 1
                error(f"[{i}/{total}] ✗ {key}: {result['error']}")
            else:
                success += 1
                log(f"[{i}/{total}] ✓ {key} | bibs={result['bibs']} faces={result['faces']}")

        log(f"Complete: {success} success, {failed} failed")

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
