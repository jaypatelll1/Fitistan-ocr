"""
Local Docker Test - Bib Number Extraction
Pass image path to test OCR without S3
"""

import os
import json
import sys

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from paddleocr import PaddleOCR

# OCR (created once, reused for all calls)
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
            use_gpu=False,
            show_log=False
        )
    return ocr


def process_image(image_path):
    """Process a local image and extract bib numbers"""

    if not os.path.exists(image_path):
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Image not found: {image_path}'})
        }

    # Run OCR on the image
    result = get_ocr().ocr(image_path, cls=True)

    # Extract only numbers (bib numbers)
    bibs = []
    if result and result[0]:
        for detection in result[0]:
            text = detection[1][0].strip()
            if text.isdigit():
                bibs.append(text)

    print(f"Image: {image_path} | Bibs found: {bibs}")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'image': image_path,
            'bibs': bibs,
            'count': len(bibs)
        })
    }


if __name__ == '__main__':
    # Run from command line: python lambda_handler.py /path/to/image.jpg
    if len(sys.argv) < 2:
        print("Usage: python lambda_handler.py <image_path>")
        print("Example: python lambda_handler.py /images/test.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    result = process_image(image_path)
    print(json.dumps(json.loads(result['body']), indent=2))
