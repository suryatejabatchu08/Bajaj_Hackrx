from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
import requests
import tempfile
import fitz  # PyMuPDF for PDF parsing
import uuid
import os
from google.generativeai import configure, GenerativeModel, embed_content
from sentence_transformers import SentenceTransformer  # Kept in case fallback needed
from pinecone import Pinecone
from dotenv import load_dotenv
from PyPDF2 import PdfReader
load_dotenv()

app = FastAPI()

# =============================
# Configuration..
# =============================

# Set your Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=GEMINI_API_KEY)
llm_model = GenerativeModel("gemini-1.5-flash")
embedding_model_name = "models/embedding-001"  # This produces 768-dim vectors

# Pinecone
PINECONE_INDEX_NAME = "hackrx"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if index exists and create it if it doesn't
def ensure_index_exists():
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        return index
    except Exception:
        # Index doesn't exist, create it with correct dimensions
        from pinecone import ServerlessSpec
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Match the Gemini embedding-001 model
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait a moment for the index to be ready
        import time
        time.sleep(10)
        return pc.Index(PINECONE_INDEX_NAME)

index = ensure_index_exists()

# =============================
# Models
# =============================

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: Dict

# =============================
# Main Endpoint
# =============================

@app.post("/api/v1/hackrx/run", response_model=Dict[str, List[Answer]])
def run_query(request: QueryRequest):
    print("[DEBUG] Received request with documents:", request.documents)
    print("[DEBUG] Questions:", request.questions)
    # Clear existing data to ensure fresh start with correct metadata
    try:
        print("[DEBUG] Clearing Pinecone index...")
        index.delete(delete_all=True)
        print("[DEBUG] Pinecone index cleared.")
    except Exception as e:
        print(f"[DEBUG] Exception while clearing index: {e}")
        pass  # Index might be empty or not exist
    
    print("[DEBUG] Extracting and chunking PDF...")
    text_chunks = extract_and_chunk_pdf(request.documents)
    print(f"[DEBUG] Extracted {len(text_chunks)} chunks from PDF.")
    print("[DEBUG] Indexing chunks...")
    index_chunks(text_chunks)
    print("[DEBUG] Chunks indexed.")

    results = []
    for question in request.questions:
        print(f"[DEBUG] Processing question: {question}")
        matched_clauses = retrieve_similar_chunks(question)
        print(f"[DEBUG] Retrieved {len(matched_clauses)} similar chunks.")
        decision = evaluate_with_gemini(question, matched_clauses)
        print(f"[DEBUG] Decision for question: {decision}")
        results.append({"question": question, "answer": decision})

    print("[DEBUG] Returning results.")
    return {"answers": results}

def fetch_and_parse_pdf(url):
    print(f"[DEBUG] Attempting to download PDF from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        print("[DEBUG] PDF downloaded successfully.")
        reader = PdfReader("temp.pdf")
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        print("[DEBUG] PDF parsed successfully.")
        return text
    except Exception as e:
        print(f"[DEBUG] Error fetching/parsing PDF: {e}")
        return None
# =============================
# PDF Parsing
# =============================

def extract_and_chunk_pdf(url: str) -> List[Dict]:
    print(f"[DEBUG] Downloading PDF for chunking from: {url}")
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    print(f"[DEBUG] PDF saved to temporary file: {tmp_path}")
    doc = fitz.open(tmp_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        print(f"[DEBUG] Extracted text from page {page_num+1}, length: {len(text)}")
        for para in text.split("\n\n"):
            if len(para.strip()) > 50:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": para.strip(),
                    "metadata": {"page": page_num + 1}
                })
    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    return chunks

# =============================
# Embedding + Indexing
# =============================

def embed_text(text: str) -> List[float]:
    print(f"[DEBUG] Embedding text of length {len(text)}")
    result = embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embedding = result["embedding"]
    print(f"[DEBUG] Embedding generated, length: {len(embedding)}")
    
    # The Gemini embedding-001 model produces 768-dimensional vectors
    # If your Pinecone index expects 1536 dimensions, you have two options:
    # 1. Recreate your Pinecone index with dimension=768
    # 2. Use a different embedding model that produces 1536 dimensions
    
    # For now, let's use the 768-dimensional embedding as-is
    # You'll need to update your Pinecone index to match this dimension
    return embedding

def index_chunks(chunks: List[Dict]) -> List[str]:
    print(f"[DEBUG] Indexing {len(chunks)} chunks...")
    ids = []
    for chunk in chunks:
        print(f"[DEBUG] Indexing chunk ID: {chunk['id']}")
        embedding = embed_text(chunk["text"])
        ids.append(chunk["id"])
        # Store both the original metadata and the text content
        metadata = chunk["metadata"].copy()
        metadata["text"] = chunk["text"]
        index.upsert([(chunk["id"], embedding, metadata)])
    print(f"[DEBUG] Finished indexing chunks.")
    return ids

# =============================
# Retrieval
# =============================

def retrieve_similar_chunks(query: str, top_k: int = 5) -> List[Dict]:
    print(f"[DEBUG] Retrieving similar chunks for query: {query}")
    query_embedding = embed_text(query)
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    print(f"[DEBUG] Retrieved {len(response.matches)} matches from Pinecone.")
    return response.matches

# =============================
# Gemini LLM Evaluation
# =============================

def evaluate_with_gemini(question: str, matches: List[Dict]) -> Dict:
    print(f"[DEBUG] Evaluating with Gemini for question: {question}")
    clauses = []
    for i, match in enumerate(matches):
        page = match.metadata.get('page', 'Unknown')
        text = match.metadata.get('text', 'No text available')
        clauses.append(f"[{i+1}] Page {page}: {text}")
    clauses_text = "\n\n".join(clauses)
    print(f"[DEBUG] Constructed clauses text for Gemini prompt, length: {len(clauses_text)}")
    prompt = f"""
You are a legal assistant AI. Based on the question and the matched clauses, respond in this JSON format:
{{
  "decision": "Yes/No/Partially",
  "rationale": "Explanation of the decision",
  "conditions": ["Condition 1", "Condition 2"],
  "clause_references": [1, 2]
}}

Question: {question}

Matched Clauses:
{clauses_text}
"""
    print(f"[DEBUG] Sending prompt to Gemini LLM...")
    response = llm_model.generate_content(prompt)
    print(f"[DEBUG] Gemini LLM response: {response.text}")
    try:
        import json
        import re
        raw_response = response.text.strip()
        print(f"[DEBUG] Raw Gemini response: {repr(raw_response)}")
        
        # Try to extract JSON from markdown code blocks
        # Pattern to match content between ```json and ``` or just ``` and ```
        json_pattern = r'```(?:json)?\s*\n([\s\S]*?)\n```'
        match = re.search(json_pattern, raw_response)
        
        if match:
            response_clean = match.group(1).strip()
            print(f"[DEBUG] Extracted JSON from markdown block: {repr(response_clean)}")
        else:
            # If no markdown block found, try to find JSON in the raw response
            # Look for content that starts with { and ends with }
            json_content_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_content_pattern, raw_response)
            if json_match:
                response_clean = json_match.group(0)
                print(f"[DEBUG] Extracted JSON content: {repr(response_clean)}")
            else:
                response_clean = raw_response
                print(f"[DEBUG] Using raw response as JSON: {repr(response_clean)}")
        
        parsed = json.loads(response_clean)
        print(f"[DEBUG] Parsed Gemini response: {parsed}")
        return parsed
    except Exception as e:
        print(f"[DEBUG] Error parsing Gemini response: {e}")
        return {"decision": "Unknown", "rationale": "Parsing failed", "conditions": [], "clause_references": []}