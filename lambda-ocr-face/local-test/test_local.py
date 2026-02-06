"""
Image Processing Pipeline - OCR + Face Detection
Downloads image from S3, extracts bib numbers, detects faces, saves to PostgreSQL
"""
import os
import sys
import json
import tempfile
import logging
from typing import Optional
from dataclasses import dataclass

import cv2
import boto3
import numpy as np
import psycopg
from PIL import Image
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set PaddleOCR environment before import
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis


@dataclass
class Config:
    """Application configuration from environment variables"""
    # Database
    db_host: str = os.getenv('DB_HOST', '')
    db_port: int = int(os.getenv('DB_PORT', '5432'))
    db_name: str = os.getenv('DB_NAME', '')
    db_user: str = os.getenv('DB_USER', '')
    db_password: str = os.getenv('DB_PASSWORD', '')
    db_sslmode: str = os.getenv('DB_SSLMODE', 'require')

    # AWS
    aws_access_key: str = os.getenv('AWS_ACCESS_KEY_ID', '')
    aws_secret_key: str = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    aws_region: str = os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')

    # S3
    s3_bucket: str = os.getenv('S3_BUCKET', 'fitistan-image-processing')

    # Processing
    max_image_size: tuple = (1200, 1200)
    face_det_size: tuple = (1024, 1024)
    face_det_thresh: float = 0.4
    ocr_det_limit: int = 1280


config = Config()


class DatabaseManager:
    """Handles all database operations"""

    def __init__(self):
        self.conn: Optional[psycopg.Connection] = None

    def connect(self):
        """Establish database connection"""
        if self.conn is None or self.conn.closed:
            logger.info("Connecting to database...")
            self.conn = psycopg.connect(
                host=config.db_host,
                port=config.db_port,
                dbname=config.db_name,
                user=config.db_user,
                password=config.db_password,
                sslmode=config.db_sslmode
            )
            register_vector(self.conn)
            logger.info("Database connected")
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")

    def save_bib_numbers(self, image_name: str, bibs: list, url: str):
        """Save extracted bib numbers to database"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_photo_bib_number (image_name, bib_number, url, event_code)
                VALUES (%s, %s, %s, NULL)
                """,
                (image_name, json.dumps(bibs), url)
            )
        conn.commit()
        logger.info(f"Saved bib numbers for {image_name}: {bibs}")

    def save_face_embedding(self, image_name: str, face_index: int,
                           embedding: list, url: str, event_code: str = None):
        """Save face embedding to database"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_face_embedding (image_name, face_index, embedding, event_code, url)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (image_name, face_index) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    event_code = EXCLUDED.event_code,
                    url = EXCLUDED.url,
                    last_modified = now()
                """,
                (image_name, face_index, embedding, event_code, url)
            )
        conn.commit()
        logger.info(f"Saved face {face_index} embedding for {image_name}")


class S3Manager:
    """Handles S3 operations"""

    def __init__(self):
        self.client = boto3.client(
            's3',
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_secret_key,
            region_name=config.aws_region
        )

    def download_file(self, bucket: str, key: str, local_path: str):
        """Download file from S3"""
        logger.info(f"Downloading s3://{bucket}/{key}")
        self.client.download_file(bucket, key, local_path)

    def list_images(self, bucket: str, prefix: str = "", max_keys: int = 100) -> list:
        """List images in S3 bucket"""
        logger.info(f"Listing images in s3://{bucket}/{prefix}")

        response = self.client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys
        )

        images = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(key)

        return images

    def get_url(self, bucket: str, key: str) -> str:
        """Generate S3 URL for an object"""
        return f"https://{bucket}.s3.{config.aws_region}.amazonaws.com/{key}"


class OCRProcessor:
    """Handles OCR operations for bib number extraction"""

    def __init__(self):
        self._model: Optional[PaddleOCR] = None

    @property
    def model(self) -> PaddleOCR:
        """Lazy load OCR model"""
        if self._model is None:
            logger.info("Loading PaddleOCR model...")
            self._model = PaddleOCR(
                lang='ch',
                ocr_version='PP-OCRv4',
                use_angle_cls=True,
                use_gpu=False,
                show_log=False,
                use_textline_orientation=True,
                det_limit_side_len=config.ocr_det_limit,
                text_det_thresh=0.1,
                text_det_box_thresh=0.1,
                text_rec_score_thresh=0.1,
            )
            logger.info("PaddleOCR model loaded")
        return self._model

    def extract_bib_numbers(self, image_path: str) -> list:
        """Extract bib numbers (digits only) from image"""
        logger.info("Running OCR...")
        result = self.model.ocr(image_path, cls=True)

        bibs = []
        if result and result[0]:
            for detection in result[0]:
                text = detection[1][0].strip()
                confidence = detection[1][1]
                logger.debug(f"  Detected: '{text}' (confidence: {confidence:.2f})")
                if text.isdigit():
                    bibs.append(text)

        logger.info(f"Found {len(bibs)} bib numbers: {bibs}")
        return bibs


class FaceProcessor:
    """Handles face detection and embedding extraction"""

    def __init__(self):
        self._model: Optional[FaceAnalysis] = None

    @property
    def model(self) -> FaceAnalysis:
        """Lazy load face model"""
        if self._model is None:
            logger.info("Loading InsightFace model...")
            self._model = FaceAnalysis(name="buffalo_l")
            self._model.prepare(
                ctx_id=0,
                det_size=config.face_det_size,
                det_thresh=config.face_det_thresh
            )
            logger.info("InsightFace model loaded")
        return self._model

    def detect_faces(self, image_path: str) -> list:
        """Detect faces and extract embeddings from image"""
        logger.info("Running face detection...")

        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return []

        faces = self.model.get(img)
        logger.info(f"Detected {len(faces)} faces")

        results = []
        for index, face in enumerate(faces, start=1):
            embedding = face.embedding.astype(np.float32).tolist()
            results.append({
                'index': index,
                'embedding': embedding,
                'bbox': face.bbox.astype(int).tolist()
            })
            logger.debug(f"  Face {index}: embedding dim = {len(embedding)}")

        return results


class ImageProcessor:
    """Main image processing pipeline"""

    def __init__(self):
        self.s3 = S3Manager()
        self.db = DatabaseManager()
        self.ocr = OCRProcessor()
        self.face = FaceProcessor()

    def resize_image(self, input_path: str, output_path: str):
        """Resize image for processing"""
        with Image.open(input_path) as img:
            img.thumbnail(config.max_image_size, Image.Resampling.LANCZOS)
            img.save(output_path)

    def process(self, bucket: str, key: str) -> dict:
        """Process a single image from S3"""
        logger.info(f"Processing: s3://{bucket}/{key}")

        # Create temp files
        original_path = tempfile.mktemp(suffix='.jpg')
        resized_path = tempfile.mktemp(suffix='_resized.jpg')

        try:
            # Download from S3
            self.s3.download_file(bucket, key, original_path)

            # Resize for processing
            self.resize_image(original_path, resized_path)

            # Extract bib numbers
            bibs = self.ocr.extract_bib_numbers(resized_path)

            # Detect faces
            faces = self.face.detect_faces(resized_path)

            # Build URL and extract filename
            url = self.s3.get_url(bucket, key)
            filename = os.path.basename(key)  # e.g., "JayTest1.jpg"

            # Save to database
            self.db.save_bib_numbers(key, bibs, url)  # bib table uses full path

            for face_data in faces:
                self.db.save_face_embedding(
                    image_name=filename,  # face table uses just filename
                    face_index=face_data['index'],
                    embedding=face_data['embedding'],
                    url=url
                )

            result = {
                'status': 'success',
                'image': key,
                'bibs': bibs,
                'faces_count': len(faces)
            }

            logger.info(f"Completed: {key} - {len(bibs)} bibs, {len(faces)} faces")
            return result

        except Exception as e:
            logger.error(f"Error processing {key}: {e}")
            return {
                'status': 'error',
                'image': key,
                'error': str(e)
            }

        finally:
            # Cleanup temp files
            for path in [original_path, resized_path]:
                if os.path.exists(path):
                    os.remove(path)

    def close(self):
        """Cleanup resources"""
        self.db.close()


def main():
    """Main entry point"""
    processor = ImageProcessor()

    bucket = config.s3_bucket

    # Handle command line arguments
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        prefix = sys.argv[2] if len(sys.argv) >= 3 else "test-images/"
        images = processor.s3.list_images(bucket, prefix)
        print(f"\nImages in s3://{bucket}/{prefix}:")
        for img in images:
            print(f"  {img}")
        return

    # Default test image
    key = "test-images/JayTest1.jpg"

    if len(sys.argv) >= 3:
        bucket = sys.argv[1]
        key = sys.argv[2]
    elif len(sys.argv) == 2:
        key = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"Bucket: {bucket}")
    print(f"Key: {key}")
    print(f"{'='*60}\n")

    try:
        result = processor.process(bucket, key)
        print(f"\n{'='*60}")
        print(f"Result: {json.dumps(result, indent=2)}")
        print(f"{'='*60}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()