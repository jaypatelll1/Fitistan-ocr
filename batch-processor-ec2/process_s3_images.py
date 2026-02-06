"""
Batch S3 Image Processor
Fetches 10 images from S3 and logs bib numbers found via OCR
"""

import os
import tempfile
import boto3

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from paddleocr import PaddleOCR

# Configuration - from environment variables or defaults
S3_BUCKET = os.environ.get('S3_BUCKET', 'fitistan-image-processing')
S3_PREFIX = os.environ.get('S3_PREFIX', 'raw-images/test/')
MAX_IMAGES = int(os.environ.get('MAX_IMAGES', '10'))

# S3 client
s3 = boto3.client('s3')

# OCR instance (lazy loaded)
ocr = None


def get_ocr():
    """Get or create OCR instance"""
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            lang='ch',
            ocr_version='PP-OCRv4',
            use_textline_orientation=True,
            text_det_thresh=0.1,
            text_det_box_thresh=0.1,
            text_rec_score_thresh=0.1,
        )
    return ocr


def get_image_keys(bucket, prefix, max_count):
    """List image keys from S3 bucket"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    keys = []

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(image_extensions):
                keys.append(key)
                if len(keys) >= max_count:
                    return keys
    return keys


def extract_bibs(image_path):
    """Run OCR and extract bib numbers (digits only)"""
    result = get_ocr().predict(image_path)
    bibs = []

    if result and len(result) > 0:
        rec_texts = result[0].get('rec_texts', [])
        rec_scores = result[0].get('rec_scores', [])

        for text, score in zip(rec_texts, rec_scores):
            text = text.strip()
            if text.isdigit() and len(text) >= 2 and score > 0.5:
                bibs.append(text)

    return bibs


def process_images():
    """Main function to process S3 images"""
    print(f"Fetching up to {MAX_IMAGES} images from s3://{S3_BUCKET}/{S3_PREFIX}")
    print("-" * 50)

    # Get image keys
    image_keys = get_image_keys(S3_BUCKET, S3_PREFIX, MAX_IMAGES)

    if not image_keys:
        print("No images found!")
        return

    print(f"Found {len(image_keys)} images\n")

    # Process each image
    for i, key in enumerate(image_keys, 1):
        ext = os.path.splitext(key)[1] or '.jpg'
        tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            # Download from S3
            s3.download_file(S3_BUCKET, key, tmp_path)

            # Extract bib numbers
            bibs = extract_bibs(tmp_path)

            # Log results
            print(f"[{i}/{len(image_keys)}] {key}")
            if bibs:
                print(f"    Bibs found: {bibs}")
            else:
                print(f"    No bib numbers found")

        except Exception as e:
            print(f"[{i}/{len(image_keys)}] {key}")
            print(f"    ERROR: {e}")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("-" * 50)
    print("Done!")


if __name__ == '__main__':
    process_images()