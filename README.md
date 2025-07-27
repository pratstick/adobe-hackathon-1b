# Persona-Driven Document Intelligence (Challenge 1B)

## Overview
This solution addresses Challenge 1B of the Adobe India Hackathon, focusing on Persona-Driven Document Intelligence. The system acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done. It integrates the outline generation (from Challenge 1A) directly into its processing pipeline, ensuring a seamless and self-contained execution.

## Approach
The core of this solution is a modular "Retrieve & Rank" pipeline, designed to semantically understand user intent and extract highly relevant document sections. The pipeline consists of the following stages:

1.  **Integrated 1A Processing (Outline Generation)**: Before persona-driven analysis, the system first processes all input PDFs to extract their structured outlines (titles, H1, H2, H3 headings) using a YOLO-based model. This outline data is generated in-memory and used to define precise section boundaries within the documents.
    **Optimization:** The YOLO model now processes all pages of a PDF in a single batch, significantly reducing inference time for multi-page documents.

2.  **Ingestion & Chunking**: PDFs are processed using `PyMuPDF` to extract text. The in-memory structured outlines are used to define precise section boundaries. Each identified section, along with its text content, document name, page number, and title, forms a "document chunk" in our corpus.

3.  **Query Formulation**: The user's high-level need, expressed through the `Persona` and `Job-to-be-Done`, is translated into a single, potent semantic query. This is achieved by concatenating the persona and job descriptions.

4.  **Corpus & Query Encoding**: A pre-trained Sentence Transformer model (`tomaarsen/static-similarity-mrl-multilingual-v1`) is used to convert both the formulated query and every document chunk in the corpus into high-dimensional vector embeddings. This model is specifically chosen for its efficiency on CPU, small size, and suitability for asymmetric search tasks.
    **Model Selection Rationale:** The `static-similarity-mrl-multilingual-v1` model is optimized for resource-constrained environments and uses static embeddings for faster performance, making it ideal for containerized deployments with limited computational resources.

5.  **Retrieval & Ranking**: With the query and corpus represented as vectors in the same semantic space, a vector similarity search is performed using `sentence_transformers.util.semantic_search`. This function efficiently calculates the similarity between the query vector and all corpus vectors, retrieving the top-k most similar chunks and ranking them by their relevance score.

6.  **Output Generation**: The final stage assembles the processed information into the precise JSON structure required by the challenge specification. This includes `metadata`, `extracted_sections`, and `subsection_analysis` (document, page number, and the most relevant paragraph from the section).

## Models and Libraries Used
*   **PyMuPDF (fitz)**: For efficient and accurate text extraction from PDF documents.
*   **sentence-transformers**: The primary library for generating semantic embeddings. The specific model used is `tomaarsen/static-similarity-mrl-multilingual-v1` (≈ 430MB), chosen for resource-constrained environments and faster static embedding performance.
*   **torch**: The underlying deep learning framework for `sentence-transformers` 
*   **ultralytics**: Used for object detection within PDFs to identify titles and section headers (YOLOv11s DocLayNet model ≈ 19MB, Ultralytics library ≈ 60MB).
*   **Pillow**: Used for image processing (part of the integrated 1A processing; library ≈ 3MB).
*   **onnxruntime**: Used for optimized ONNX model inference (library ≈ 30MB).

## How to Build and Run

### Prerequisites
*   Docker installed on your system.


### 1. Build the Docker Image
Build the Docker image from the project root:

```bash
docker build --platform linux/amd64 -t persona-driven-doc-intel:latest .
```


### 2. Run the Docker Container

### Expected Input Structure

Before running the pipeline, ensure your input files are organized as follows:

- Place all PDF documents to be processed in the directory: `input/PDFs/`
- Place your input JSON file (containing persona and job-to-be-done information) at: `input/input.json`
- Create an empty `output` directory in the project root to store results.


Your project directory should look like this:

```
input/
  input.json         # The main input JSON file
  PDFs/              # Directory containing all input PDF files
models/
  static-similarity-mrl-multilingual-v1/   # Pre-downloaded SentenceTransformer model (≈ 430MB)
output/              # Output directory for results
```

#### Model Storage and Versioning

The model `static-similarity-mrl-multilingual-v1` is stored in the `models/` directory and tracked using [Git Large File Storage (LFS)](https://git-lfs.com/). This helps avoid repeated downloads and keeps the repository size manageable.

> ℹ**Note:** Git LFS is used to version and store this model. If you're cloning this repo, make sure Git LFS is installed to access the model files correctly.

If the model is not present in `models/`, the Docker build will automatically download it as a fallback.

### Running the Docker Container

Run the Docker container, mounting the `input` and `output` directories:

```bash
docker run --rm \
  -v "$(pwd)"/input:/app/input \
  -v "$(pwd)"/output:/app/output \
  --network none \
  persona-driven-doc-intel:latest \
  python process_pdfs_1b.py /app/input/your_input.json /app/output
```

**Example for Collection 1:**

First, prepare your input directory. For `round_1b_002`, you would have `challenge1b_input.json` and a `PDFs` subdirectory containing the PDF documents.

```bash
mkdir -p input/PDFs
mkdir -p output
# cp Challenge_1b/documents/challenge1b_input.json input/
# cp Challenge_1b/documents/PDFs/* input/PDFs/
```

Then run the Docker command:

```bash
docker run --rm \
  -v "$(pwd)"/input:/app/input \
  -v "$(pwd)"/output:/app/output \
  --network none \
  persona-driven-doc-intel:latest \
  python process_pdfs_1b.py /app/input/challenge1b_input.json /app/output
```

### 3. View Results
The generated output JSON file (e.g., `round_1b_002_travel_planner_output.json`) will be available in the `output` directory.
