"""
AWS Lambda - Image Processing Pipeline
Extracts bib numbers (OCR) and face embeddings from S3 images
"""
import os
import json
import tempfile

# Set environment before imports (Lambda filesystem is read-only except /tmp)
os.environ['HOME'] = '/tmp'
os.environ['PADDLEX_HOME'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import boto3
import numpy as np
import psycopg
from PIL import Image
from urllib.parse import unquote_plus
from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis
from pgvector.psycopg import register_vector

# Configuration
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_SSLMODE = os.environ.get('DB_SSLMODE', 'require')
AWS_REGION = os.environ.get('AWS_DEFAULT_REGION', 'ap-south-1')

# Lazy-loaded models
_ocr = None
_face_model = None
_s3 = boto3.client('s3')


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            lang='ch',
            ocr_version='PP-OCRv4',
            use_angle_cls=True,
            use_gpu=False,
            show_log=False,
            det_limit_side_len=1280,
        )
    return _ocr


def get_face_model():
    global _face_model
    if _face_model is None:
        _face_model = FaceAnalysis(name="buffalo_l")
        _face_model.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.4)
    return _face_model


def get_db_connection():
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE
    )
    register_vector(conn)
    return conn


def resize_image(input_path, output_path, max_size=1200):
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img.save(output_path)


def extract_bib_numbers(image_path):
    result = get_ocr().ocr(image_path, cls=True)
    bibs = []
    if result and result[0]:
        for detection in result[0]:
            text = detection[1][0].strip()
            if text.isdigit():
                bibs.append(text)
    return bibs


def extract_face_embeddings(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []

    faces = get_face_model().get(img)
    embeddings = []
    for idx, face in enumerate(faces, start=1):
        embeddings.append({
            'index': idx,
            'embedding': face.embedding.astype(np.float32).tolist()
        })
    return embeddings


def save_bib_numbers(conn, image_name, bibs, url):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO event_photo_bib_number (image_name, bib_number, url, event_code)
            VALUES (%s, %s, %s, NULL)
            ON CONFLICT (image_name) DO UPDATE SET
                bib_number = EXCLUDED.bib_number,
                url = EXCLUDED.url,
                last_modified = now()
            """,
            (image_name, json.dumps(bibs), url)
        )
    conn.commit()


def save_face_embedding(conn, image_name, face_index, embedding, url):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO event_face_embedding (image_name, face_index, embedding, event_code, url)
            VALUES (%s, %s, %s, NULL, %s)
            ON CONFLICT (image_name, face_index) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                url = EXCLUDED.url,
                last_modified = now()
            """,
            (image_name, face_index, embedding, url)
        )
    conn.commit()


def lambda_handler(event, context):
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = unquote_plus(record['s3']['object']['key'])

    original_path = tempfile.mktemp(suffix='.jpg')
    resized_path = tempfile.mktemp(suffix='_resized.jpg')

    try:
        # Download from S3
        _s3.download_file(bucket, key, original_path)
        resize_image(original_path, resized_path)

        # Process image
        bibs = extract_bib_numbers(resized_path)
        faces = extract_face_embeddings(resized_path)

        # Build URL and filenames
        url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{key}"
        filename = os.path.basename(key)

        # Save to database
        conn = get_db_connection()
        try:
            save_bib_numbers(conn, key, bibs, url)
            for face in faces:
                save_face_embedding(conn, filename, face['index'], face['embedding'], url)
        finally:
            conn.close()

        print(f"Processed: {key} | Bibs: {bibs} | Faces: {len(faces)}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'image': key,
                'bibs': bibs,
                'faces_count': len(faces)
            })
        }

    except Exception as e:
        print(f"Error: {key} - {e}")
        raise

    finally:
        for path in [original_path, resized_path]:
            if os.path.exists(path):
                os.remove(path)
