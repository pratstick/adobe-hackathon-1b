# Approach Explanation: Persona-Driven Document Intelligence

## Overview
This solution addresses Challenge 1B of the Adobe India Hackathon, focusing on Persona-Driven Document Intelligence. The system acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done. It integrates the outline generation (from Challenge 1A) directly into its processing pipeline, ensuring a seamless and self-contained execution.

## Methodology
The core of this solution is a modular "Retrieve & Rank" pipeline, designed to semantically understand user intent and extract highly relevant document sections. The pipeline consists of the following stages:

1.  **Integrated 1A Processing (Outline Generation)**: Before persona-driven analysis, the system first processes all input PDFs to extract their structured outlines (titles, H1, H2, H3 headings) using a YOLO-based model. This outline data is generated in-memory and used to define precise section boundaries within the documents.

2.  **Ingestion & Chunking**: PDFs are processed using `PyMuPDF` to extract text. The in-memory structured outlines are used to define precise section boundaries. Each identified section, along with its text content, document name, page number, and title, forms a "document chunk" in our corpus.

3.  **Query Formulation**: The user's high-level need, expressed through the `Persona` and `Job-to-be-Done`, is translated into a single, potent semantic query. This is achieved by concatenating the persona and job descriptions.

4.  **Corpus & Query Encoding**: A pre-trained Sentence Transformer model (`static-similarity-mrl-multilingual-v1`) is used to convert both the formulated query and every document chunk in the corpus into high-dimensional vector embeddings. This model is specifically chosen for its efficiency on CPU, small size, and suitability for asymmetric search tasks.

5.  **Retrieval & Ranking**: With the query and corpus represented as vectors in the same semantic space, a vector similarity search is performed using `sentence_transformers.util.semantic_search`. This function efficiently calculates the similarity between the query vector and all corpus vectors, retrieving the top-k most similar chunks and ranking them by their relevance score.

6.  **Output Generation**: The final stage assembles the processed information into the precise JSON structure required by the challenge specification. This includes `metadata`, `extracted_sections`, and `subsection_analysis` (document, page number, and the most relevant paragraph from the section).

## Models and Libraries Used
*   **PyMuPDF (fitz)**: For efficient and accurate text extraction from PDF documents.
*   **sentence-transformers**: The primary library for generating semantic embeddings. The specific model used is `static-similarity-mrl-multilingual-v1`.
*   **torch**: The underlying deep learning framework for `sentence-transformers`.
*   **ultralytics**: Used for object detection within PDFs to identify titles and section headers.
*   **Pillow**: Used for image processing.
*   **onnxruntime**: Used for optimized ONNX model inference.