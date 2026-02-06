#!/usr/bin/env python3
"""
Local Test - Image Processor
Test OCR and face detection on local or S3 images.

Usage:
    python test_processor.py image.jpg
    python test_processor.py ./images/
    python test_processor.py s3://bucket/prefix/ --max 5
    python test_processor.py image.jpg --save-db
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Load .env file if exists
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")


class Processor:
    def __init__(self):
        self._ocr = None
        self._face = None

    @property
    def ocr(self):
        if self._ocr is None:
            print("Loading OCR model...")
            self._ocr = PaddleOCR(
                lang="ch",
                ocr_version="PP-OCRv4",
                use_angle_cls=True,
                use_gpu=False,
                show_log=False,
            )
        return self._ocr

    @property
    def face(self):
        if self._face is None:
            print("Loading face model...")
            self._face = FaceAnalysis(name="buffalo_l")
            self._face.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.4)
        return self._face

    def extract_bibs(self, image_path: str) -> list:
        result = self.ocr.ocr(image_path, cls=True)
        bibs = []
        if result and result[0]:
            for det in result[0]:
                text = det[1][0].strip()
                if text.isdigit():
                    bibs.append(text)
        return bibs

    def extract_faces(self, image_path: str) -> list:
        img = cv2.imread(image_path)
        if img is None:
            return []
        faces = self.face.get(img)
        return [
            {"index": i + 1, "embedding": f.embedding.astype(np.float32).tolist()}
            for i, f in enumerate(faces)
        ]


class S3Client:
    def __init__(self):
        self.client = boto3.client("s3", region_name=AWS_REGION)

    def list_images(self, bucket: str, prefix: str, max_count: int = None) -> list:
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = os.path.splitext(key)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    keys.append(key)
                    if max_count and len(keys) >= max_count:
                        return keys
        return keys

    def download(self, bucket: str, key: str, local_path: str):
        self.client.download_file(bucket, key, local_path)


class Database:
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

    def save_bibs(self, image_name: str, bibs: list, url: str):
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

    def save_face(self, image_name: str, face_index: int, embedding: list, url: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_face_embedding (image_name, face_index, embedding, event_code, url)
                VALUES (%s, %s, %s, NULL, %s)
                ON CONFLICT (image_name, face_index) DO UPDATE SET
                    embedding = EXCLUDED.embedding, url = EXCLUDED.url, last_modified = now()
                """,
                (image_name, face_index, embedding, url),
            )
        self.conn.commit()

    def close(self):
        self.conn.close()


def parse_s3_path(s3_path: str) -> tuple:
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def get_local_images(path: str) -> list:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(p.glob(f"*{ext}"))
            images.extend(p.glob(f"*{ext.upper()}"))
        return [str(img) for img in sorted(images)]
    return []


def main():
    parser = argparse.ArgumentParser(description="Test OCR and face detection")
    parser.add_argument("path", help="Local path or S3 path (s3://bucket/prefix/)")
    parser.add_argument("--max", type=int, help="Max images (for S3)")
    parser.add_argument("--save-db", action="store_true", help="Save results to database")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    is_s3 = args.path.startswith("s3://")
    processor = Processor()
    results = []
    db = None

    if args.save_db:
        print("Connecting to database...")
        db = Database()

    try:
        if is_s3:
            # S3 mode
            bucket, prefix = parse_s3_path(args.path)
            s3 = S3Client()
            keys = s3.list_images(bucket, prefix, args.max)
            print(f"Found {len(keys)} image(s) in S3\n")

            for i, key in enumerate(keys, 1):
                name = os.path.basename(key)
                print(f"[{i}/{len(keys)}] Processing: {name}")

                ext = os.path.splitext(key)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    s3.download(bucket, key, tmp_path)
                    bibs = processor.extract_bibs(tmp_path)
                    faces = processor.extract_faces(tmp_path)

                    result = {"image": key, "bibs": bibs, "faces": len(faces)}
                    results.append(result)
                    print(f"    Bibs: {bibs} | Faces: {len(faces)}")

                    if db:
                        url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{key}"
                        db.save_bibs(key, bibs, url)
                        for face in faces:
                            db.save_face(name, face["index"], face["embedding"], url)
                        print("    ✓ Saved to DB")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                print()

        else:
            # Local mode
            images = get_local_images(args.path)
            if not images:
                print(f"No images found: {args.path}")
                sys.exit(1)

            print(f"Found {len(images)} image(s)\n")

            for i, img_path in enumerate(images, 1):
                name = os.path.basename(img_path)
                print(f"[{i}/{len(images)}] Processing: {name}")

                bibs = processor.extract_bibs(img_path)
                faces = processor.extract_faces(img_path)

                result = {"image": name, "bibs": bibs, "faces": len(faces)}
                results.append(result)
                print(f"    Bibs: {bibs} | Faces: {len(faces)}\n")

        if args.json:
            print("\n--- JSON Output ---")
            print(json.dumps(results, indent=2))

        print(f"Done! Processed {len(results)} image(s)")

    finally:
        if db:
            db.close()


if __name__ == "__main__":
    main()
