"""
Local test script for bib number extraction
Contains all code for: S3 download, OCR, PostgreSQL insert
"""
import os
import json
import tempfile
import boto3
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set PaddleOCR environment before import
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from paddleocr import PaddleOCR

# Database connection
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')

# S3 Client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_DEFAULT_REGION', 'ap-south-1')
)

# OCR instance
ocr = None


def get_ocr():
    """Get or create OCR instance"""
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            lang='ch',
            ocr_version='PP-OCRv4',
            use_angle_cls=True,
            use_gpu=False,
            show_log=False,
            use_textline_orientation=True,
            det_limit_side_len=1280,
            text_det_thresh=0.1,
            text_det_box_thresh=0.1,
            text_rec_score_thresh=0.1,
        )
    return ocr


def save_to_database(image_name, bibs, url):
    """Save bib numbers to PostgreSQL database"""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO event_photo_bib_number (image_name, bib_number, url, event_code)
            VALUES (%s, %s, %s, NULL)
            """,
            (image_name, json.dumps(bibs), url)
        )
        conn.commit()
        cursor.close()
        print(f"Saved to database: {image_name}")
    except Exception as e:
        print(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def process_image(bucket: str, key: str):
    """Download image from S3, run OCR, save to database"""

    # Download image to temp file
    tmp_path = tempfile.mktemp(suffix='.jpg')
    print(f"Downloading s3://{bucket}/{key}...")
    s3.download_file(bucket, key, tmp_path)

    try:
        # Run OCR
        print("Running OCR...")
        result = get_ocr().ocr(tmp_path, cls=True)

        # Extract only numbers (bib numbers)
        bibs = []
        if result and result[0]:
            for detection in result[0]:
                text = detection[1][0].strip()
                confidence = detection[1][1]
                print(f"  Detected: '{text}' (confidence: {confidence:.2f})")
                if text.isdigit():
                    bibs.append(text)

        print(f"\nImage: {key} | Bibs found: {bibs}")

        # Save to database
        save_to_database(key, bibs, key)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'image': key,
                'bibs': bibs,
                'count': len(bibs)
            })
        }

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def list_images(bucket: str, prefix: str = "", max_keys: int = 20):
    """List images in S3 bucket"""
    print(f"Listing images in s3://{bucket}/{prefix}...")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"  {obj['Key']}")
    else:
        print("  No images found")


if __name__ == "__main__":
    import sys

    bucket = "fitistan-image-processing"

    # List images if --list flag
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        prefix = sys.argv[2] if len(sys.argv) >= 3 else "test-images/"
        list_images(bucket, prefix)
        sys.exit(0)

    # Default test values
    key = "test-images/JayTest1.jpg"

    # Allow command line override
    if len(sys.argv) >= 3:
        bucket = sys.argv[1]
        key = sys.argv[2]
    elif len(sys.argv) == 2:
        key = sys.argv[1]

    print(f"Bucket: {bucket}")
    print(f"Key: {key}")
    print("-" * 50)

    result = process_image(bucket, key)

    print("-" * 50)
    print(f"Result: {result}")