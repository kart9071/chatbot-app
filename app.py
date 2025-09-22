import os
import json
import logging
import sys
import asyncio
import uuid
from flask import Flask, request, jsonify, render_template
import boto3
from concurrent.futures import ThreadPoolExecutor

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# --- Env ---
KB_ID_1 = "TKWQDABBSG"
KB_ID_2 = "WDBONTXTWY"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
AWS_REGION = "us-east-1"
DYNAMO_TABLE = "RAGCacheTable"

# --- AWS Clients ---
agent_client = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
session_table = dynamodb.Table(DYNAMO_TABLE)

# --- Flask ---
app = Flask(__name__)

# ----------------- DynamoDB Helpers -----------------
def get_session_context(session_id: str):
    """Fetch session context from DynamoDB."""
    response = session_table.get_item(Key={"cache_key": session_id})
    return response.get("Item", {}).get("context", [])

def save_session_context(session_id: str, context: list):
    """Save session context to DynamoDB."""
    session_table.put_item(Item={"cache_key": session_id, "context": context})

def append_to_session(session_id: str, message: str):
    """Append a new message to session context in DB."""
    context = get_session_context(session_id)
    context.append(message)
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
def retrieve_from_kb(kb_id: str, query: str):
    config = {'vectorSearchConfiguration': {'numberOfResults': 7}}
    res = agent_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={'text': query},
        retrievalConfiguration=config
    )
    return res.get('retrievalResults', [])

def merge_results(a, b):
    all_docs = a + b
    all_docs.sort(key=lambda r: r.get('score', 0), reverse=True)
    return "\n\n".join([r['content']['text'] for r in all_docs])

# ----------------- Preprocessing LLM -----------------
async def preprocess_query(session_id: str, query: str):
    """Use LLM to rewrite query using conversation context."""
    previous_context = "\n".join(get_session_context(session_id))

    prompt = f"""You are a query rewriting assistant.

Conversation history:
{previous_context}

User question:
{query}

Task:
- Always rewrite the user’s question into a fully explicit version.
- Replace vague references like "this patient", "these patients", "who is severe" etc.
  with the exact patient names or IDs from the conversation history.
- If multiple patients are in context, expand them explicitly.
- If no patients are in history, leave the question unchanged.

Return ONLY the rewritten question. No explanation, no commentary."""

    def invoke():
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                "max_tokens": 300,
                "temperature": 0.2
            })
        )
        out = json.loads(response['body'].read().decode())
        return out['content'][0]['text'].strip()

    loop = asyncio.get_event_loop()
    rewritten = await loop.run_in_executor(None, invoke)

    # Save user query + rewritten
    append_to_session(session_id, f"User: {query}")
    append_to_session(session_id, f"Rewritten: {rewritten}")

    return rewritten

# ----------------- Claude Call -----------------
async def call_claude_async(query: str, context_text: str):
    loop = asyncio.get_event_loop()
    prompt = f"""
Use the following documents to answer:

Context:
{context_text}

Question:
{query}

Answer:"""

    def invoke():
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                "max_tokens": 12500,
                "temperature": 0.2
            })
        )
        out = json.loads(response['body'].read().decode())
        return out['content'][0]['text']

    return await loop.run_in_executor(None, invoke)

# ----------------- Routes -----------------
# ----------------- Routes -----------------
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

    # Step 2 - KB retrieval
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        r1_future = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_1, rewritten_query)
        r2_future = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_2, rewritten_query)
        docs1, docs2 = await asyncio.gather(r1_future, r2_future)

    # Log metadata and content for KB1
    logger.info(f"Session {session_id} | KB1 ({KB_ID_1}) retrieved {len(docs1)} documents")
    for i, doc in enumerate(docs1, 1):
        content = doc['content'].get('text', '')
        metadata = {k: v for k, v in doc['content'].items() if k != 'text'}
        logger.info(f"KB1 Doc {i} | Metadata: {metadata} | Content: {content[:500]}")  # first 500 chars

    # Log metadata and content for KB2
    logger.info(f"Session {session_id} | KB2 ({KB_ID_2}) retrieved {len(docs2)} documents")
    for i, doc in enumerate(docs2, 1):
        content = doc['content'].get('text', '')
        metadata = {k: v for k, v in doc['content'].items() if k != 'text'}
        logger.info(f"KB2 Doc {i} | Metadata: {metadata} | Content: {content[:500]}")  # first 500 chars

    # Step 3 - Merge KB results
    context_text = merge_results(docs1, docs2)

    # Step 4 - Claude answer
    answer = await call_claude_async(rewritten_query, context_text)

    # Step 5 - Save answer
    append_to_session(session_id, f"Answer: {answer}")

    return jsonify({
        "answer": answer,
        "rewritten_query": rewritten_query,
        "source_docs_count": len(docs1) + len(docs2),
        "session_id": session_id
    })

@app.route("/patients", methods=["POST"])
def patients():
    """
    Add patients to session context.
    Body example:
    {
        "session_id": "123",
        "patients": [
            {"id": 2, "name": "Timothy M Grafinger"},
            {"id": 3, "name": "Willie Goins"}
        ]
    }
    """
    body = request.json
    session_id = body.get("session_id")
    patients = body.get("patients", [])

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    if not patients:
        return jsonify({"error": "No patients provided"}), 400

    store_patients_in_context(session_id, patients)
    return jsonify({"message": "Patients stored", "patients": patients})

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
