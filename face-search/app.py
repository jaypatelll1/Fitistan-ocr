import os
import cv2
import psycopg
import numpy as np
import boto3
from PIL import Image
from insightface.app import FaceAnalysis
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

# =========================
# LOAD .env (for DB only)
# =========================
load_dotenv(override=True)

# =========================
# S3 CONFIG
# =========================
S3_BUCKET = "fitistan-image-processing"
S3_PREFIX = "raw-images/BEG/"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

AWS_REGION = "ap-south-1"
EVENT_CODE = "123214"

BASE_S3_URL = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/"

AWS_ACCESS_KEY_ID = os.getenv("AWS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# =========================
# DB CONFIG
# =========================
DB_CONFIG = {
    "host": os.getenv("NEON_HOST"),
    "port": 5432,
    "dbname": os.getenv("NEON_DB"),
    "user": os.getenv("NEON_USER"),
    "password": os.getenv("NEON_PASSWORD"),
    "sslmode": "require",
}

# =========================
# DB CONNECTION
# =========================
print("Connecting to NeonDB...")
conn = psycopg.connect(**DB_CONFIG)
register_vector(conn)
cur = conn.cursor()
print("Connected to NeonDB.")

# =========================
# INIT FACE MODEL
# =========================
def initialize_face_model():
    print("Loading InsightFace model...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.4)
    print("Model loaded.")
    return app

face_model = initialize_face_model()

# =========================
# RESIZE IMAGE
# =========================
def resize_image(input_path, output_path, max_width=1200, max_height=1200):
    with Image.open(input_path) as img:
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        img.save(output_path)

# =========================
# UPSERT FUNCTION
# =========================
def upsert_embedding(image_name, face_index, embedding, s3_key):
    image_url = BASE_S3_URL + s3_key

    cur.execute(
        """
        INSERT INTO event_face_embedding 
            (image_name, face_index, embedding, event_code, url)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (image_name, face_index)
        DO UPDATE
        SET 
            embedding = EXCLUDED.embedding,
            event_code = EXCLUDED.event_code,
            url = EXCLUDED.url,
            last_modified = now();
        """,
        (image_name, face_index, embedding, EVENT_CODE, image_url)
    )

# =========================
# PROCESS S3 IMAGE
# =========================
def process_s3_image(s3_key):
    image_name = os.path.basename(s3_key)
    print(f"\nProcessing S3 image: s3://{S3_BUCKET}/{s3_key}")

    local_path = f"/tmp/{image_name}"
    temp_resized_path = f"/tmp/temp_{image_name}"

    try:
        s3.download_file(S3_BUCKET, s3_key, local_path)

        resize_image(local_path, temp_resized_path)

        img = cv2.imread(temp_resized_path)
        faces = face_model.get(img)

        if not faces:
            return f"No face detected: {image_name}"

        print(f"Detected {len(faces)} faces in {image_name}")

        for index, face in enumerate(faces, start=1):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                print(f"Invalid crop for face {index} in {image_name}")
                continue

            embedding_vector = face.embedding.astype(np.float32).tolist()

            print(f"  Face {index}: embedding dim = {len(embedding_vector)}")

            # ✅ FIXED: pass s3_key
            upsert_embedding(image_name, index, embedding_vector, s3_key)

            print(f"  Stored face {index} for {image_name}")

        conn.commit()
        return f"Processed {image_name}"

    except Exception as e:
        conn.rollback()
        return f"Error processing {image_name}: {e}"

    finally:
        for path in [local_path, temp_resized_path]:
            if os.path.exists(path):
                os.remove(path)

# =========================
# LIST S3 IMAGES
# =========================
def list_s3_images():
    paginator = s3.get_paginator("list_objects_v2")

    image_keys = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(IMAGE_EXTENSIONS):
                image_keys.append(key)

    return image_keys

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print(f"Listing images from s3://{S3_BUCKET}/{S3_PREFIX}")

    image_keys = list_s3_images()
    print(f"Found {len(image_keys)} images in S3")

    # 🔴 SAFETY: Uncomment for testing
    # image_keys = image_keys[:10]

    for key in image_keys:
        result = process_s3_image(key)
        print(result)

    cur.close()
    conn.close()

    print("\nAll S3 images processed.")
