import os
import json
import logging
import sys
import asyncio
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
KB_ID_1 = "WDBONTXTWY"
KB_ID_2 = "TKWQDABBSG"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
AWS_REGION = "us-east-1"

# --- AWS Clients ---
agent_client = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)

# --- Flask ---
app = Flask(__name__)

# In-memory conversation history
session_contexts = {}

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

    previous_context = "\n".join(session_contexts.get(session_id, []))

    prompt = f"""You are a query reformulation assistant.
You receive user questions and previous conversation history.
Your task is to rewrite the question into a fully explicit version by replacing vague references (like "patient 1", "him", "previous result")
with full details found in the history. If no such details exist, just return the original question unchanged.

Conversation history:
{previous_context}

User question:
{query}

Return ONLY the rewritten question, nothing else. No explanation, no commentary, no extra words."""

    def invoke():
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                "max_tokens": 200,
                "temperature": 0.2
            })
        )
        out = json.loads(response['body'].read().decode())
        return out['content'][0]['text'].strip()

    loop = asyncio.get_event_loop()
    rewritten = await loop.run_in_executor(None, invoke)

    # Save only user query (not answer yet)
    session_contexts.setdefault(session_id, []).append(f"User: {query}")
    session_contexts[session_id].append(f"Rewritten: {rewritten}")

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

# ----------------- Route -----------------
@app.route("/chat", methods=["POST"])
async def chat():
    body = request.json
    query = body.get("query")
    session_id = body.get("session_id", "default")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Step 1 - Rewrite the query using context
    rewritten_query = await preprocess_query(session_id, query)
    logger.info(f"Original: {query} | Rewritten: {rewritten_query}")

    # Step 2 - Parallel KB retrieval
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        r1 = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_1, rewritten_query)
        r2 = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_2, rewritten_query)
        docs1, docs2 = await asyncio.gather(r1, r2)

    # Step 3 - Merge KB results
    context_text = merge_results(docs1, docs2)

    # Step 4 - Ask Claude with cleaned query
    answer = await call_claude_async(rewritten_query, context_text)

    # ✅ Step 5 - Save the answer into session_contexts
    session_contexts[session_id].append(f"Answer: {answer}")

    return jsonify({
        "answer": answer,
        "rewritten_query": rewritten_query,
        "source_docs_count": len(docs1) + len(docs2)
    })

# ----------------- Frontend -----------------
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
