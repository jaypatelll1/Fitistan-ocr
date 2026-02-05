"""
AWS Lambda - Bib Number Extraction
Triggered when an image is uploaded to S3
"""

import os
import json
import boto3
import tempfile
from urllib.parse import unquote_plus

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from paddleocr import PaddleOCR

# S3 Client
s3 = boto3.client('s3')

# OCR (created once, reused for all calls)
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
            text_det_thresh=0.1,
            text_det_box_thresh=0.1,
            text_rec_score_thresh=0.1,
        )
    return ocr


def lambda_handler(event, context):
    """Called automatically when an image is uploaded to S3"""

    # Get the uploaded file info from S3 event
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = unquote_plus(record['s3']['object']['key'])

    # Download image to temp file
    tmp_path = tempfile.mktemp(suffix='.jpg')
    s3.download_file(bucket, key, tmp_path)

    try:
        # Run OCR on the image
        result = get_ocr().ocr(tmp_path, cls=True)

        # Extract only numbers (bib numbers)
        bibs = []
        if result and result[0]:
            for detection in result[0]:
                text = detection[1][0].strip()
                if text.isdigit():
                    bibs.append(text)

        print(f"Image: {key} | Bibs found: {bibs}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'image': key,
                'bibs': bibs,
                'count': len(bibs)
            })
        }

    finally:
        # Always delete temp file
        os.remove(tmp_path)
