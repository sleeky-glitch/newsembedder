import streamlit as st
import os
from pdf2image import convert_from_path
import pytesseract
import openai
import pinecone
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path
import tempfile
from pinecone import ServerlessSpec

# Streamlit page config
st.set_page_config(
    page_title="PDF Text Embedder",
    page_icon="📚",
    layout="wide"
)

# Initialize OpenAI
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone with new method
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pinecone.api_key=PINECONE_API_KEY

# Create or connect to an existing index
index_name = "vectornews"
dimension = 1536
index = pinecone.index(index_name)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using OCR"""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)

        # Create progress bar
        progress_bar = st.progress(0)

        # Extract text from each image
        text_chunks = []
        for i, image in enumerate(images):
            # Update progress
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)

            # Set Tesseract language to Gujarati
            text = pytesseract.image_to_string(image, lang='guj')
            chunks = text.split('\n\n')
            text_chunks.extend([chunk.strip() for chunk in chunks if chunk.strip()])

        return text_chunks
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def get_embedding(text):
    """Generate embedding using OpenAI's text-embedding-ada-002 model"""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.warning(f"Retrying embedding generation due to error: {e}")
        time.sleep(1)
        return get_embedding(text)

def batch_embeddings(texts, batch_size=100):
    """Generate embeddings in batches to handle API rate limits"""
    all_embeddings = []

    progress_bar = st.progress(0)
    total_batches = len(range(0, len(texts), batch_size))

    for i, batch_start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[batch_start:batch_start + batch_size]
        try:
            response = openai.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(embeddings)

            # Update progress
            progress = (i + 1) / total_batches
            progress_bar.progress(progress)

            time.sleep(0.1)
        except Exception as e:
            st.warning(f"Retrying batch with smaller size due to error: {e}")
            time.sleep(1)
            return batch_embeddings(texts, batch_size=batch_size//2)

    return all_embeddings

def process_pdf_and_store(pdf_path, file_name):
    """Process PDF and store embeddings in Pinecone"""
    try:
        with st.spinner("Extracting text from PDF..."):
            text_chunks = extract_text_from_pdf(pdf_path)

        if not text_chunks:
            st.error("No text extracted from PDF")
            return

        with st.spinner("Generating embeddings..."):
            embeddings = batch_embeddings(text_chunks)

        with st.spinner("Storing embeddings in Pinecone..."):
            batch_size = 100
            file_prefix = Path(file_name).stem

            total_batches = len(range(0, len(text_chunks), batch_size))
            progress_bar = st.progress(0)

            for i in range(0, len(text_chunks), batch_size):
                batch_texts = text_chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]

                vectors = [
                    {
                        'id': f'{file_prefix}_chunk_{i+j}',
                        'values': emb,
                        'metadata': {
                            'text': text,
                            'source': file_name,
                            'chunk_id': i+j,
                            'processed_date': time.strftime('%Y-%m-%d')
                        }
                    }
                    for j, (text, emb) in enumerate(zip(batch_texts, batch_embeddings))
                ]

                index.upsert(vectors=vectors)

                # Update progress
                progress = (i + batch_size) / len(text_chunks)
                progress_bar.progress(min(progress, 1.0))

        st.success(f"Successfully processed {file_name}")

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

def main():
    st.title("📚 PDF Text Embedder")
    st.write("Upload Gujarati PDF files to extract text and store embeddings in Pinecone")

    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Number of files uploaded: {len(uploaded_files)}")

        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                process_pdf_and_store(tmp_file_path, uploaded_file.name)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        st.success("All files processed successfully!")

if __name__ == "__main__":
    main()
