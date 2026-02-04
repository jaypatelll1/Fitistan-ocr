"""
Bib Number Extractor - CPU Version
Uses ch_PP-OCRv4_server_rec model optimized for bib detection
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import csv
from datetime import datetime

# ========== CONFIGURATION ==========
IMAGE_DIR = r'C:\Users\Admin\Desktop\Jay-Image_processing\Image processing\Done_compressed'

# Model settings (best for bib detection)
MODEL = 'ch_PP-OCRv4_server_rec'
LANGUAGE = 'ch'

# Detection settings (optimized for bib numbers)
DET_DB_THRESH = 0.1
DET_DB_BOX_THRESH = 0.1
DROP_SCORE = 0.1
DET_LIMIT_SIDE_LEN = 1280
# ===================================

from paddleocr import PaddleOCR

# Initialize OCR (CPU mode)
ocr = PaddleOCR(
    lang=LANGUAGE,
    use_angle_cls=True,
    use_gpu=False,
    show_log=False,
    rec_model_dir=MODEL,
    det_db_thresh=DET_DB_THRESH,
    det_db_box_thresh=DET_DB_BOX_THRESH,
    drop_score=DROP_SCORE,
    det_limit_side_len=DET_LIMIT_SIDE_LEN,
)

print(f"\n{'='*50}")
print(f"Bib Extractor (CPU) - {MODEL}")
print(f"{'='*50}\n")

# Get all images
images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"Found {len(images)} images\n")

# Process and store results
results = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i, image_name in enumerate(images, 1):
    image_path = os.path.join(IMAGE_DIR, image_name)
    print(f"[{i}/{len(images)}] {image_name}...", end=' ')

    try:
        result = ocr.ocr(image_path, cls=True)

        bib_numbers = []
        if result and result[0]:
            for detection in result[0]:
                text = detection[1][0].strip()
                if text.isdigit():
                    bib_numbers.append(text)

        results.append({
            'image_name': image_name,
            'bib_numbers': ', '.join(bib_numbers),
            'count': len(bib_numbers)
        })

        if bib_numbers:
            print(f"Found: {', '.join(bib_numbers)}")
        else:
            print("No bibs found")

    except Exception as e:
        print(f"Error: {str(e)[:30]}")
        results.append({
            'image_name': image_name,
            'bib_numbers': '',
            'count': 0
        })

# Save to CSV
output_file = os.path.join(IMAGE_DIR, f'bibs_cpu_{timestamp}.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['image_name', 'bib_numbers', 'count'])
    writer.writeheader()
    writer.writerows(results)

# Summary
total_bibs = sum(r['count'] for r in results)
images_with_bibs = sum(1 for r in results if r['count'] > 0)

print(f"\n{'='*50}")
print(f"DONE!")
print(f"{'='*50}")
print(f"Images processed: {len(images)}")
print(f"Images with bibs: {images_with_bibs}")
print(f"Total bibs found: {total_bibs}")
print(f"Output: {output_file}")
print(f"{'='*50}\n")
