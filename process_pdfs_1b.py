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

def _process_yolo_results_for_page(page, yolo_result, model_names):
    page_headings = []
    page_title = ""
    page_text_blocks = page.get_text("dict")["blocks"]

    for box in yolo_result.boxes:
        class_name = model_names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        if class_name == 'Title':
            page_title = page.get_text("text", clip=(x1, y1, x2, y2)).strip()
        elif class_name == 'Section-header':
            detected_text = ""
            font_size = 0.0
            is_bold = False
            x0 = 0.0
            
            for block in page_text_blocks:
                if block['type'] == 0: # Text block
                    for line in block['lines']:
                        for span in line['spans']:
                            span_x0, span_y0, span_x1, span_y1 = span['bbox']
                            
                            span_center_x = (span_x0 + span_x1) / 2
                            span_center_y = (span_y0 + span_y1) / 2
                            
                            if x1 <= span_center_x <= x2 and y1 <= span_center_y <= y2:
                                detected_text += span['text'] + " "
                                if span['size'] > font_size:
                                    font_size = span['size']
                                is_bold = bool(span['flags'] & 16)
                                x0 = span['bbox'][0]
            
            if detected_text:
                page_headings.append({
                    "level": "Section-header",
                    "text": detected_text.strip(),
                    "page": page.number, # Use page.number for 0-based index
                    "y1": y1,
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "x0": x0
                })
    return page_headings, page_title

# --- 1A Logic (Integrated) ---
def load_yolo_model(model_path):
    return YOLO(model_path)

def process_pdf_1a(pdf_file_path, yolo_model):
    doc = fitz.open(pdf_file_path)
    all_headings = []
    model_detected_title = ""

    page_images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(img)

    yolo_results = yolo_model.predict(page_images, save=False, conf=0.5)

    for page_num, page in enumerate(doc):
        page_yolo_result = yolo_results[page_num]
        page_headings, page_title = _process_yolo_results_for_page(page, page_yolo_result, yolo_model.names)
        all_headings.extend(page_headings)
        if page_title and not model_detected_title:
            model_detected_title = page_title

    all_headings.sort(key=lambda x: (x['page'], x['y1']))
    classified_headings = classify_headings(all_headings)
    final_title = determine_title(classified_headings, model_detected_title)

    for h in classified_headings:
        if 'y1' in h:
            del h['y1']
        if 'font_size' in h:
            del h['font_size']
        if 'is_bold' in h:
            del h['is_bold']
        if 'x0' in h:
            del h['x0']

    return {"title": final_title, "outline": classified_headings}



def classify_headings(headings):
    section_header_candidates = [h for h in headings if h['level'] == 'Section-header']
    section_header_candidates.sort(key=lambda x: (x['font_size'], x['is_bold'], -x['x0']), reverse=True)

    assigned_levels = {}
    current_h_level = 1

    for candidate in section_header_candidates:
        feature_key = (candidate['font_size'], candidate['is_bold'], candidate['x0'])
        
        if feature_key not in assigned_levels:
            if current_h_level <= 3:
                assigned_levels[feature_key] = f'H{current_h_level}'
                current_h_level += 1
            else:
                assigned_levels[feature_key] = 'H3'
        
        candidate['level'] = assigned_levels[feature_key]
    
    return headings

def determine_title(headings, model_detected_title):
    title = model_detected_title
    if not title and headings:
        candidate_titles = [h for h in headings if h['level'] in ['H1', 'H2', 'H3']]
        if candidate_titles:
            candidate_titles.sort(key=lambda x: (x['page'], int(x['level'][1]), x['y1']))
            title = candidate_titles[0]['text']
    return title

def process_pdf_1a(pdf_file_path, yolo_model):
    doc = fitz.open(pdf_file_path)
    all_headings = []
    model_detected_title = ""

    page_images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(img)

    yolo_results = yolo_model.predict(page_images, save=False, conf=0.5)

    for page_num, page in enumerate(doc):
        page_yolo_result = yolo_results[page_num]
        page_headings, page_title = _process_yolo_results_for_page(page, page_yolo_result, yolo_model.names)
        all_headings.extend(page_headings)
        if page_title and not model_detected_title:
            model_detected_title = page_title

    all_headings.sort(key=lambda x: (x['page'], x['y1']))
    classified_headings = classify_headings(all_headings)
    final_title = determine_title(classified_headings, model_detected_title)

    for h in classified_headings:
        if 'y1' in h:
            del h['y1']
        if 'font_size' in h:
            del h['font_size']
        if 'is_bold' in h:
            del h['is_bold']
        if 'x0' in h:
            del h['x0']

    return {"title": final_title, "outline": classified_headings}

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
    sub_section_analysis = []

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

            sub_section_analysis.append({
                "document": original_section['doc_name'],
                "page_number": original_section['page'],
                "refined_text": refined_text
            })

    return extracted_sections, sub_section_analysis

def generate_final_output(pdf_names, persona, job, extracted_sections, sub_section_analysis, output_path):
    final_output = {
        "metadata": {
            "input_documents": pdf_names,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated output at {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_pdfs_1b.py <input_pdf_dir> <output_json_dir> <persona> <job_to_be_done>")
        sys.exit(1)

    input_pdf_dir = Path(sys.argv[1])
    output_json_dir = Path(sys.argv[2])
    persona = sys.argv[3]
    job_to_be_done = sys.argv[4]

    output_json_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = [f for f in input_pdf_dir.glob("*.pdf") if f.is_file()]
    pdf_names = [f.name for f in pdf_files]

    # Load YOLO model for 1A processing
    yolo_model = load_yolo_model('/home/pratyush/repos/adobe-hackathon-1b/models/yolov12m-doclaynet.pt')

    # Perform 1A processing for all PDFs and store outlines in memory
    outline_data_map = {}
    for pdf_file in pdf_files:
        print(f"Processing 1A for {pdf_file.name}...")
        outline_data = process_pdf_1a(pdf_file, yolo_model)
        outline_data_map[pdf_file.name] = outline_data

    # 1. Ingestion and Content Preparation (using in-memory outline_data_map)
    corpus = create_corpus_from_pdfs(pdf_files, outline_data_map)

    # 2. Semantic Encoding and Model Selection
    model = SentenceTransformer('tomaarsen/static-similarity-mrl-multilingual-v1')
    query_embedding, corpus_embeddings = encode_query_and_corpus(persona, job_to_be_done, corpus, model)

    # 3. Retrieval, Ranking, and Sub-Section Analysis
    extracted_sections, sub_section_analysis = retrieve_and_rank_sections(query_embedding, corpus_embeddings, corpus, model)

    # 4. Final Output Generation
    output_file_path = output_json_dir / "challenge1b_output.json"
    generate_final_output(pdf_names, persona, job_to_be_done, extracted_sections, sub_section_analysis, output_file_path)