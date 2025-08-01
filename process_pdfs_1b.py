
import json
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import torch
import datetime
import re
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import sys
import unicodedata


# --- Updated 1A Logic ---
def _combine_text_blocks(text_blocks, min_y=0, max_y=400):
    """
    Combines multi-line text blocks within a vertical range (y) into a single string.
    Useful for extracting multi-line titles or headings.
    """
    lines = []
    for block in text_blocks:
        if block['type'] == 0: # Text block
            for line in block['lines']:
                for span in line['spans']:
                    y0 = span['bbox'][1]
                    if min_y <= y0 <= max_y:
                        text = unicodedata.normalize('NFKC', span['text']).strip()
                        if text:
                            lines.append(text)
    # Combine lines, preserving order
    return ' '.join(lines)

def _process_yolo_results_for_page(page, yolo_result, model_names):
    """
    Processes YOLO results for a single page to extract headings and title.
    """
    page_headings = []
    page_title = ""
    page_text_blocks = page.get_text("dict")["blocks"]

    for box in yolo_result.boxes:
        class_name = model_names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if class_name == 'Title':
            page_title = unicodedata.normalize('NFKC', page.get_text(clip=(x1, y1, x2, y2)).strip())
        elif class_name == 'Section-header':
            spans = []
            for block in page_text_blocks:
                if block['type'] == 0: # Text block
                    for line in block['lines']:
                        for span in line['spans']:
                            span_x0, span_y0, span_x1, span_y1 = span['bbox']
                            if not (x2 < span_x0 or x1 > span_x1 or y2 < span_y0 or y1 > span_y1):
                                spans.append(span)
            if spans:
                spans.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
                merged_text = ""
                last_x1 = -1
                for span in spans:
                    if last_x1 != -1 and span['bbox'][0] > last_x1 + 1:
                        merged_text += " "
                    merged_text += span['text']
                    last_x1 = span['bbox'][2]
                detected_text = unicodedata.normalize('NFKC', merged_text.strip())
                font_size = max(s['size'] for s in spans)
                is_bold = any(s['flags'] & 16 for s in spans)
                x0 = spans[0]['bbox'][0]
                text_case = _get_text_case(detected_text)
                page_headings.append({
                    "level": "Section-header",
                    "text": detected_text,
                    "page": page.number,
                    "y1": y1,
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "x0": x0,
                    "text_case": text_case
                })
    return page_headings, page_title

# --- 1A Logic (Integrated) ---
def load_yolo_model(model_path):
    return YOLO(model_path)

def _extract_page_text_and_image(page):
    """
    Extracts image and raw text blocks from a single page.
    """
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    page_text_blocks = page.get_text("dict")["blocks"]
    return img, page_text_blocks

def _get_text_case(text):
    if text.isupper():
        return "upper"
    elif text.istitle():
        return "title"
    else:
        return "sentence"

def process_pdf_1a(pdf_file_path, yolo_model):
    doc = fitz.open(pdf_file_path)
    all_headings = []
    model_detected_title = ""
    first_page_text_info = []
    all_font_sizes = []

    page_images = []
    page_text_blocks_list = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img, text_blocks = _extract_page_text_and_image(page)
        page_images.append(img)
        page_text_blocks_list.append(text_blocks)
        
        for block in text_blocks:
            if block['type'] == 0: # Text block
                for line in block['lines']:
                    for span in line['spans']:
                        all_font_sizes.append(span['size'])
                        if page_num == 0:
                            first_page_text_info.append({
                                "text": unicodedata.normalize('NFKC', span['text']).strip(),
                                "bbox": span['bbox'],
                                "font_size": span['size'],
                                "is_bold": bool(span['flags'] & 16)
                            })

    yolo_results = yolo_model.predict(page_images, save=False, conf=0.5)

    for page_num, page in enumerate(doc):
        page_yolo_result = yolo_results[page_num]
        page_headings, page_title = _process_yolo_results_for_page(page, page_yolo_result, yolo_model.names)
        all_headings.extend(page_headings)
        if page_title and not model_detected_title:
            model_detected_title = page_title

    all_headings.sort(key=lambda x: (x['page'], x['y1']))

    # Calculate font thresholds
    if all_font_sizes:
        unique_font_sizes = sorted(list(set(all_font_sizes)), reverse=True)
        # Simple heuristic: largest font size is H1, second largest is H2, etc.
        # This can be made more sophisticated with statistical methods if needed.
        font_thresholds = {
            "H1": unique_font_sizes[0] if len(unique_font_sizes) > 0 else 0,
            "H2": unique_font_sizes[1] if len(unique_font_sizes) > 1 else 0,
            "H3": unique_font_sizes[2] if len(unique_font_sizes) > 2 else 0,
        }
    else:
        font_thresholds = {"H1": 0, "H2": 0, "H3": 0}

    classified_headings = classify_headings(all_headings, font_thresholds)
    final_title = determine_title(classified_headings, model_detected_title, first_page_text_info)

    # Clean up temporary attributes used for processing
    for h in classified_headings:
        if 'y1' in h:
            del h['y1']
        if 'font_size' in h:
            del h['font_size']
        if 'is_bold' in h:
            del h['is_bold']
        if 'x0' in h:
            del h['x0']
        if 'text_case' in h:
            del h['text_case']

    return {"title": final_title, "outline": classified_headings}


def classify_headings(headings, font_thresholds):
    """
    Dynamically assigns heading levels (H1, H2, H3, ...) based on sorted unique stylistic properties.
    Supports more than three heading levels for longer documents.
    """
    section_header_candidates = [h for h in headings if h['level'] == 'Section-header']
    if not section_header_candidates:
        return headings
    style_groups = {}
    for h in section_header_candidates:
        style_key = (h['font_size'], h['is_bold'], h['x0'], h['text_case'])
        if style_key not in style_groups:
            style_groups[style_key] = []
        style_groups[style_key].append(h)
    case_priority = {"upper": 3, "title": 2, "sentence": 1}
    sorted_unique_styles = sorted(
        style_groups.keys(),
        key=lambda x: (-x[0], -x[1], x[2], -case_priority.get(x[3], 0))
    )
    # Dynamically assign heading levels: H1, H2, H3, ...
    level_map = {}
    for i, style_key in enumerate(sorted_unique_styles):
        level_map[style_key] = f"H{i+1}"
    for h in section_header_candidates:
        style_key = (h['font_size'], h['is_bold'], h['x0'], h['text_case'])
        h['level'] = level_map.get(style_key, f"H{len(sorted_unique_styles)}")
    return headings


def determine_title(headings, model_detected_title, first_page_text_info):
    """
    Determines the document title using model detection and a robust fallback mechanism
    that analyzes text blocks from the first page. More robust heuristics for longer, lower, and larger titles.
    """
    # 1. Prioritize model_detected_title if it's clean and reasonable
    if model_detected_title and model_detected_title.strip():
        # Allow longer titles, relax word count, and allow some newlines if not excessive
        word_count = len(model_detected_title.split())
        if word_count >= 2 and word_count <= 30 and model_detected_title.count("\n") < 3:
            return model_detected_title.replace("\n", " ").strip()
    # 2. Fallback to heuristic-based title extraction from first page text info
    if first_page_text_info:
        potential_titles = []
        for text_info in first_page_text_info:
            word_count = len(text_info['text'].split())
            is_title_candidate = (
                (text_info['is_bold'] and text_info['font_size'] > 12) or
                (text_info['font_size'] >= 22)
            ) and text_info['bbox'][1] < 400 and 2 <= word_count <= 35
            if is_title_candidate:
                potential_titles.append(text_info)
        if potential_titles:
            # Sort by font size (desc), boldness (desc), y-position (asc)
            potential_titles.sort(key=lambda x: (x['font_size'], x['is_bold'], -x['bbox'][1]), reverse=True)
            # Combine all candidate title blocks within y-threshold
            # Use _combine_text_blocks to merge multi-line title
            # Use the y-threshold of 0-400 for title region
            combined_title = _combine_text_blocks([
                {
                    'type': 0,
                    'lines': [
                        {'spans': [
                            {'text': t['text'], 'bbox': t['bbox']}
                        ]}
                    ]
                } for t in potential_titles
            ], min_y=0, max_y=400)
            if combined_title:
                return combined_title
    # 3. Fallback to the first H1 heading or the first heading in the document
    if headings:
        first_page_headings = [h for h in headings if h['page'] == 0]
        if first_page_headings:
            # Sort by heading level (H1, H2, H3, ...), then by y-position
            first_page_headings.sort(key=lambda x: (int(x['level'][1:]), x['y1']))
            return first_page_headings[0]['text']
        headings.sort(key=lambda x: (x['page'], int(x['level'][1:]), x['y1']))
        return headings[0]['text']
    # 4. Last resort: return the first non-empty text block from the first page
    if first_page_text_info:
        for text_info in first_page_text_info:
            if text_info['text'].strip():
                return text_info['text'].strip()
    return ""

# --- 1B Logic ---
def create_corpus_from_pdfs(pdf_paths, outline_data_map):
    corpus = []
    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        outline_data = outline_data_map.get(pdf_name)

        if not outline_data:
            print(f"Warning: No outline data for {pdf_name}. Skipping.")
            continue

        doc = fitz.open(pdf_path)
        outline = sorted(outline_data.get('outline', []), key=lambda x: x.get('page', 0))

        for idx, heading in enumerate(outline):
            page_num = heading.get('page', 1) - 1
            if page_num < doc.page_count:
                section_text = doc.load_page(page_num).get_text()
                corpus.append({
                    'doc_name': pdf_name,
                    'page': heading.get('page', 0),
                    'section_title': heading.get('text', ''),
                    'section_text': section_text.strip()
                })
    return corpus

def encode_query_and_corpus(persona, job, corpus, model):
    query_text = f"Persona: {persona}. Job: {job}"
    print(f"Encoding query: {query_text}")
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    corpus_texts = [chunk['section_text'] for chunk in corpus]
    print(f"Encoding {len(corpus_texts)} corpus sections...")
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)

    return query_embedding, corpus_embeddings

def retrieve_and_rank_sections(query_embedding, corpus_embeddings, corpus, model, top_k=10):
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get results for the first (and only) query

    extracted_sections = []
    subsection_analysis = []

    print(f"Found {len(hits)} relevant sections.")

    for rank, hit in enumerate(hits):
        corpus_id = hit['corpus_id']
        original_section = corpus[corpus_id]

        extracted_sections.append({
            "document": original_section['doc_name'],
            "page_number": original_section['page'],
            "section_title": original_section['section_title'],
            "importance_rank": rank + 1
        })

        if rank < 5:
            full_section_text = original_section['section_text']
            paragraphs = re.split(r'\n\s*\n', full_section_text) 
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            refined_text = ""
            if paragraphs:
                # Limit paragraphs for refinement to the first 5 for optimization
                paragraphs_to_consider = paragraphs[:5]
                paragraph_embeddings = model.encode(paragraphs_to_consider, convert_to_tensor=True)
                paragraph_hits = util.semantic_search(query_embedding, paragraph_embeddings, top_k=1)
                if paragraph_hits and paragraph_hits[0]:
                    best_paragraph_id = paragraph_hits[0][0]['corpus_id']
                    refined_text = paragraphs_to_consider[best_paragraph_id]
                    # Remove common bullet point characters and leading whitespace
                    refined_text = re.sub(r'^[\s\t]*[*•-][\s\t]*', '', refined_text, flags=re.MULTILINE)

            subsection_analysis.append({
                "document": original_section['doc_name'],
                "page_number": original_section['page'],
                "refined_text": refined_text
            })

    return extracted_sections, subsection_analysis

def generate_final_output(pdf_names, persona, job, extracted_sections, subsection_analysis, output_path):
    final_output = {
        "metadata": {
            "input_documents": pdf_names,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated output at {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_pdfs_1b.py <input_json_path> <output_dir>")
        sys.exit(1)

    input_json_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    challenge_id = input_data["challenge_info"]["challenge_id"]
    test_case_name = input_data["challenge_info"]["test_case_name"]
    persona_role = input_data["persona"]["role"]
    job_task = input_data["job_to_be_done"]["task"]
    
    # Assuming PDFs are in the same directory as the input JSON for now, or a specified 'documents' path
    # For now, let's assume the PDFs are in a 'documents' subdirectory relative to the input JSON's directory
    pdf_base_dir = input_json_path.parent / "PDFs" 

    pdf_files_info = input_data["documents"]
    pdf_files = []
    pdf_names = []
    for doc_info in pdf_files_info:
        filename = doc_info["filename"]
        pdf_path = pdf_base_dir / filename
        if pdf_path.is_file():
            pdf_files.append(pdf_path)
            pdf_names.append(filename)
        else:
            print(f"Warning: Document not found: {pdf_path}. Skipping.")

    if not pdf_files:
        print("No valid PDF documents found to process. Exiting.")
        sys.exit(0)

    # Load YOLO model for 1A processing
    yolo_model = load_yolo_model('models/yolov11s-doclaynet.pt')

    # Perform 1A processing for all PDFs and store outlines in memory
    outline_data_map = {}
    for pdf_file in pdf_files:
        print(f"Processing 1A for {pdf_file.name}...")
        outline_data = process_pdf_1a(pdf_file, yolo_model)
        outline_data_map[pdf_file.name] = outline_data

    # 1. Ingestion and Content Preparation (using in-memory outline_data_map)
    corpus = create_corpus_from_pdfs(pdf_files, outline_data_map)

    # 2. Semantic Encoding and Model Selection
    model = SentenceTransformer('/app/models/static-similarity-mrl-multilingual-v1')
    query_embedding, corpus_embeddings = encode_query_and_corpus(persona_role, job_task, corpus, model)

    # 3. Retrieval, Ranking, and Sub-Section Analysis
    extracted_sections, subsection_analysis = retrieve_and_rank_sections(query_embedding, corpus_embeddings, corpus, model)

    # 4. Final Output Generation
    output_file_name = f"{challenge_id}_{test_case_name}_output.json"
    output_file_path = output_dir / output_file_name
    generate_final_output(pdf_names, persona_role, job_task, extracted_sections, subsection_analysis, output_file_path)