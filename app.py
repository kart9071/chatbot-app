import os
import json
import logging
import sys
import asyncio
import uuid
from flask import Flask, request, jsonify, render_template
import boto3
from concurrent.futures import ThreadPoolExecutor
import tiktoken 

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# --- Env ---
KB_ID_1 = "I5E7NJH4NE"   
KB_ID_2 = "ID51DYIDMY"
MODEL_ID = "arn:aws:bedrock:us-east-1:626635427336:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
AWS_REGION = "us-east-1"
DYNAMO_TABLE = "RAGCacheTable"
S3_BUCKET = "aadi-vidura-cache"
s3_client = boto3.client("s3", region_name=AWS_REGION)


MAX_CONTEXT_TOKENS = 3000

# --- AWS Clients ---
agent_client = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
session_table = dynamodb.Table(DYNAMO_TABLE)

# --- Flask ---
app = Flask(__name__)

def upload_qa_to_s3(session_id: str, question: str, answer: str):
    """
    Save each question-answer as a separate TXT file in S3.
    Filename format: <session_id>_<uuid>.txt
    """
    file_id = str(uuid.uuid4())
    s3_key = f"queries/{session_id}_{file_id}.txt"
    content = f"Question:\n{question}\n\nAnswer:\n{answer}"

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=content,
            ContentType="text/plain"
        )
        logger.info(f"Saved Q&A as TXT in S3: {s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload Q&A to S3: {e}")


# ----------------- DynamoDB Helpers -----------------
def get_session_context(session_id: str):
    """Fetch session context from DynamoDB."""
    response = session_table.get_item(Key={"cache_key": session_id})
    return response.get("Item", {}).get("context", [])

def save_session_context(session_id: str, context: list):
    """Save session context to DynamoDB."""
    session_table.put_item(Item={"cache_key": session_id, "context": context})

def count_tokens(text: str) -> int:
    # Approx fallback: 1 token ≈ 4 chars
    return len(text) // 4  

def append_to_session(session_id: str, message: str, min_keep: int = 5):
    """Append new message with max token sliding window but always keep last N entries."""
    context = get_session_context(session_id)
    context.append(message)

    # Flatten context into one string
    combined = "\n".join(context)

    # Trim until within token limit but keep last min_keep
    while count_tokens(combined) > MAX_CONTEXT_TOKENS and len(context) > min_keep:
        context.pop(0)   # remove oldest
        combined = "\n".join(context)

    save_session_context(session_id, context)


def delete_session(session_id: str):
    """Delete session context from DB."""
    session_table.delete_item(Key={"cache_key": session_id})

def store_patients_in_context(session_id: str, patients: list):
    """
    patients: list of dicts with keys {id, name}
    Example: [{"id": 2, "name": "Timothy M Grafinger"}, {"id": 3, "name": "Willie Goins"}]
    """
    line = "Patients in context: " + ", ".join(
        [f"{p['name']} (ID {p['id']})" for p in patients]
    )
    append_to_session(session_id, line)

# ----------------- KB Retrieval -----------------
def retrieve_from_kb(kb_id: str, query: str, search_type: str = "HYBRID"):
    config = {
        "vectorSearchConfiguration": {
            "numberOfResults": 50,
            "overrideSearchType": search_type.upper()  # ✅ correct field name
        }
    }
    res = agent_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query},
        retrievalConfiguration=config
    )
    return res.get("retrievalResults", [])

def merge_results(a, b):
    all_docs = a + b
    all_docs.sort(key=lambda r: r.get('score', 0), reverse=True)
    return "\n\n".join([r['content']['text'] for r in all_docs])

def extract_chunk_metadata(doc):
    """
    Safely extract patient metadata from a KB chunk
    """
    patient_name = "Unknown"
    patient_id = "Unknown"
    source = "Unknown"
    score = doc.get("score")

    content = doc.get("content", {})
    meta = doc.get("metadata", {})
    
    # 1. Try metadataAttributes
    attrs = meta.get("metadataAttributes", {})
    if attrs:
        patient_name = attrs.get("patient_name", patient_name)
        patient_id = attrs.get("patient_id", patient_id)

    # 2. Fallback keys
    if patient_name == "Unknown":
        patient_name = meta.get("patient_name", patient_name)
    if patient_id == "Unknown":
        patient_id = meta.get("patient_id", patient_id)

    # 3. Source
    source = meta.get("x-amz-bedrock-kb-source-uri", source)

    return {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "source": source,
        "score": score
    }

# ----------------- Preprocessing LLM -----------------
async def preprocess_query(session_id: str, query: str):
    """Let Claude decide whether to use previous context or not."""
    previous_context = "\n".join(get_session_context(session_id))

    prompt = f"""You are a query rewriting assistant.

Conversation history:
{previous_context}

User question:
{query}

Task:
1. Decide if the user’s question depends on previous conversation history.
   - If it uses vague references like "this patient", "these patients", "the last one", "he/she", 
     then you MUST rewrite the question by expanding them with the exact patient names or IDs 
     from the conversation history.
   - If the question is complete on its own (standalone), IGNORE the conversation history 
     and just return the user’s question unchanged.
   - If there are any spelling mistake than change it and give it, don't change if it is related to any date or name like entities.

2. Return ONLY the rewritten question. Do not explain your reasoning. Do not add extra text."""

    def invoke():
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                "max_tokens": 3000,
                "temperature": 0.2
            })
        )
        out = json.loads(response['body'].read().decode())
        return out['content'][0]['text'].strip()

    loop = asyncio.get_event_loop()
    rewritten = await loop.run_in_executor(None, invoke)

    append_to_session(session_id, f"User: {query}")
    append_to_session(session_id, f"Rewritten: {rewritten}")

    return rewritten

async def call_claude_with_precise_answers(query: str, content_text: str,cache_context_text:str):
    loop = asyncio.get_event_loop()
    prompt = f"""
You are an expert medical assistant. Don't answer questions other than healthcare.

You will be given patient documents (each includes name, ID, source, and content).
if the question is normal and relative to the healthcare and dont need any data to answer that question answer it directly.if you found the answer both in content and previous context combine them and give the answer.

Content:
{content_text}

Previous Context:
{cache_context_text}

Question:
{query}

Task:
1. First, provide a clear, natural language answer to the question.
2. Then, separately provide a JSON list of sources you relied on.  
   - Each entry must have: patient_name, patient_id, source.
   - Return ONLY the list in valid JSON (no commentary).

Format exactly like this:

Answer:
<your natural language answer here>

Used_Sources:
[{{"patient_name": "...", "patient_id": "...", "source": "..."}}]
"""

    def invoke():
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "max_tokens": 60000,
                "temperature": 0.1,
                "top_p":0.3,
                "top_k":50
            })
        )
        out = json.loads(response['body'].read().decode())
        return out['content'][0]['text']

    result_text = await loop.run_in_executor(None, invoke)

    # --- Split answer and sources ---
    answer_text = ""
    used_sources = []

    if "Used_Sources:" in result_text:
        parts = result_text.split("Used_Sources:")
        answer_text = parts[0].replace("Answer:", "").strip()
        json_part = parts[1].strip()

        if json_part.startswith("```"):
            json_part = json_part.strip("` \n")
            if json_part.lower().startswith("json"):
                json_part = json_part[4:].strip()

        try:
            used_sources = json.loads(json_part)
        except Exception as e:
            logger.error(f"Failed parsing used_sources JSON: {e}\nRaw: {json_part}")
            used_sources = []
    else:
        answer_text = result_text.strip()

    return answer_text, used_sources


@app.route("/chat", methods=["POST"])
async def chat():
    body = request.json
    query = body.get("query")
    session_id = body.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())
        save_session_context(session_id, [])

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Step 1 - Rewrite query
    rewritten_query = await preprocess_query(session_id, query)
    logger.info(f"Session {session_id} | Original: {query} | Rewritten: {rewritten_query}")

    # Step 2 - KB retrieval (only KB1)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        kb1_future = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_1, rewritten_query)
        kb2_future = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_2, rewritten_query)
        docs1, docs2 = await asyncio.gather(kb1_future, kb2_future)
        # r2_future = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_2, rewritten_query)
        # docs1, docs2 = await asyncio.gather(r1_future, r2_future)
    docs2 = docs2[:5] if docs2 else []

    # Step 3 - Build context text
    content_chunks = []
    for doc in docs1 + docs2:
        c = doc['content'].get('text', '')
        m = extract_chunk_metadata(doc)
        chunk_text = f"""
Patient Name: {m['patient_name']}
Patient ID: {m['patient_id']}
Source: {m['source']}

Content:
{c}
"""
        content_chunks.append(chunk_text)
    content_text = "\n\n".join(content_chunks)

    cache_context_text = "\n\n".join([doc["content"].get("text", "") for doc in docs2])
    # Step 4 - Call Claude
    answer_text, used_sources = await call_claude_with_precise_answers(rewritten_query, content_text,cache_context_text)

    logger.info(used_sources)

    # Step 5 - Save in session
    append_to_session(session_id, f"Answer: {answer_text}")

    upload_qa_to_s3(session_id,rewritten_query,answer_text)
    # Step 6 - Return response
    return jsonify({
        "answer": answer_text,
        "rewritten_query": rewritten_query,
        "source_docs_count": len(docs1),   # only KB1 now
        "used_sources": used_sources,
        "session_id": session_id
    })


@app.route("/end_session", methods=["POST"])
def end_session():
    body = request.json
    session_id = body.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    delete_session(session_id)
    return jsonify({"message": f"Session {session_id} deleted"})

# ----------------- Frontend -----------------
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
