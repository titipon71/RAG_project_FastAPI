import os
import requests
import json

import fitz

from typhoon_ocr import ocr_document


def classify_page(page,
                  min_text_chars=80,
                  large_image_ratio=0.5):
    text = page.get_text().strip()
    text_len = len(text)
    images = page.get_images(full=True)
    page_area = page.rect.width * page.rect.height

    max_img_ratio = 0
    for img in images:
        xref = img[0]
        for r in page.get_image_rects(xref):
            ratio = (r.width * r.height) / page_area
            max_img_ratio = max(max_img_ratio, ratio)

    if text_len < min_text_chars and max_img_ratio >= large_image_ratio:
        return "scanned"   # อย่างหน้า 249
    elif text_len >= min_text_chars and max_img_ratio < large_image_ratio:
        return "text"
    else:
        return "mixed"

def classify_pdf(path):
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        page_type = classify_page(page) 
        pages.append({
            "pdf_page_index": i,
            "pdf_page_number": i + 1,
            "type": page_type,
        })
    doc.close()
    return pages

result = classify_pdf("student_manual2568_1.pdf")

scanned_pages = [p for p in result if p["type"] == "scanned"]
markdown = ocr_document(
    pdf_or_image_path="student_manual2568_1.pdf",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model='scb10x/typhoon-ocr1.5-3b',
    page_num=249
    )

print(markdown)





# def extract_text_from_image(image_path, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages=None):
#     url = "https://api.opentyphoon.ai/v1/ocr"

#     with open(image_path, 'rb') as file:
#         files = {'file': file}
#         data = {
#             'model': model,
#             'task_type': task_type,
#             'max_tokens': str(max_tokens),
#             'temperature': str(temperature),
#             'top_p': str(top_p),
#             'repetition_penalty': str(repetition_penalty)
#         }

#         if pages:
#             data['pages'] = json.dumps(pages)

#         headers = {
#             'Authorization': f'Bearer {api_key}'
#         }

#         response = requests.post(url, files=files, data=data, headers=headers)

#         if response.status_code == 200:
#             result = response.json()

#             # Extract text from successful results
#             extracted_texts = []
#             for page_result in result.get('results', []):
#                 if page_result.get('success') and page_result.get('message'):
#                     content = page_result['message']['choices'][0]['message']['content']
#                     try:
#                         # Try to parse as JSON if it's structured output
#                         parsed_content = json.loads(content)
#                         text = parsed_content.get('natural_text', content)
#                     except json.JSONDecodeError:
#                         text = content
#                     extracted_texts.append(text)
#                 elif not page_result.get('success'):
#                     print(f"Error processing {page_result.get('filename', 'unknown')}: {page_result.get('error', 'Unknown error')}")

#             return '\n'.join(extracted_texts)
#         else:
#             print(f"Error: {response.status_code}")
#             print(response.text)
#             return None

# Usage
# api_key = "sk-WRDSfxT8zq4hU66Im3hKQys5geJLjR0toZd145DZoAPPGDGU"
# image_path = "student_manual2568_1.pdf"  # or path/to/your/document.pdf
# model = "typhoon-ocr"
# task_type = "default"
# max_tokens = 16000
# temperature = 0.1
# top_p = 0.6
# repetition_penalty = 1.1
# pages = 249
# extracted_text = extract_text_from_image(image_path, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages)
# print(extracted_text)