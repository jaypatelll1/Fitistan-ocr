"""
AWS Lambda - Bib Number Extraction
Processes images from S3 using PaddleOCR
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

# OCR (initialized once, reused)
ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            lang='ch',
            use_angle_cls=True,
            det_db_thresh=0.1,
            det_db_box_thresh=0.1,
            drop_score=0.1,
        )
    return ocr


def lambda_handler(event, context):
    """Main handler - processes S3 images"""
    
    results = []
    
    # Handle S3 event
    if 'Records' in event:
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])
            result = process_image(bucket, key)
            results.append(result)
    
    # Handle batch request
    elif event.get('batch'):
        bucket = event['bucket']
        prefix = event.get('prefix', '')
        
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    result = process_image(bucket, key)
                    results.append(result)
    
    # Handle single image
    elif 'bucket' in event and 'key' in event:
        result = process_image(event['bucket'], event['key'])
        results.append(result)
    
    print(f"=== RESULTS: {json.dumps(results)} ===")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': len(results),
            'results': results
        })
    }


def process_image(bucket, key):
    """Download image from S3, extract bibs"""
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        tmp_path = f.name
    s3.download_file(bucket, key, tmp_path)
    
    try:
        # Extract bib numbers
        result = get_ocr().ocr(tmp_path, cls=True)
        bibs = []
        if result and result[0]:
            for det in result[0]:
                text = det[1][0].strip()
                if text.isdigit():
                    bibs.append(text)
        
        # Log results
        print(f"Image: {key} | Bibs: {bibs}")
        
        return {'image': key, 'bibs': bibs, 'count': len(bibs)}
        
    finally:
        os.remove(tmp_path)
