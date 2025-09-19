import os
import json
import logging
import sys
import re
import asyncio
from flask import Flask, request, jsonify, render_template
import boto3
from concurrent.futures import ThreadPoolExecutor

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# --- Env Vars ---
KB_ID_1 = "WDBONTXTWY"
KB_ID_2 = "TKWQDABBSG"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
AWS_REGION = "us-east-1"

# --- AWS Clients ---
agent_client = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
comprehend = boto3.client("comprehendmedical", region_name=AWS_REGION)

# --- Flask ---
app = Flask(__name__)

# --- In-memory session storage ---
session_memory = {}

# --- Max tokens for context ---
MAX_CONTEXT_TOKENS = 15000  # adjust based on Claude model limit

# ----------------- Entity Extraction -----------------
def extract_entities(query: str):
    entities = {}
    try:
        resp = comprehend.detect_entities_v2(Text=query)
        logger.info(f"🔎 ComprehendMedical raw entities: {resp.get('Entities')}")
        for ent in resp.get("Entities", []):
            txt = ent.get("Text", "")
            cat = ent.get("Category", "")
            etype = ent.get("Type", "")
            if cat == "PROTECTED_HEALTH_INFORMATION":
                if etype == "NAME":
                    entities["patient_name"] = txt
                elif etype == "ID":
                    if "patient_id" not in entities:
                        entities["patient_id"] = txt
                    else:
                        entities["member_id"] = txt
    except Exception as e:
        logger.error(f"ComprehendMedical failed, fallback to regex: {e}")
        m = re.search(r"\bmember\s+(\w+)", query, re.I)
        if m:
            entities["member_id"] = m.group(1)
        p = re.search(r"\bpatient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", query)
        if p:
            entities["patient_name"] = p.group(1)
    return entities

# ----------------- Memory Management -----------------
def enrich_with_memory(session_id, entities, new_docs=None):
    """Keep memory of entities and KB docs across session turns."""
    if session_id not in session_memory:
        session_memory[session_id] = {"entities": {}, "knowledge_cache": {}, "last_context": ""}

    memory = session_memory[session_id]

    # Update entities
    memory["entities"].update(entities)
    for k, v in memory["entities"].items():
        if k not in entities:
            entities[k] = v

    # Update knowledge cache
    if new_docs:
        for doc in new_docs:
            doc_id = doc.get("metadata", {}).get("doc_id") or str(hash(doc['content']['text']))
            memory["knowledge_cache"][doc_id] = doc['content']['text']

    return entities

# ----------------- KB Retrieval -----------------
def retrieve_from_kb(kb_id: str, query: str, patient_ids=None):
    filter_obj = None
    if patient_ids:
        if len(patient_ids) == 1:
            filter_obj = {"equals": {"key": "patient_id", "value": patient_ids[0]}}
        elif len(patient_ids) > 1:
            filter_obj = {"in": {"key": "patient_id", "value": patient_ids}}

    config = {'vectorSearchConfiguration': {'numberOfResults': 7}}
    if filter_obj:
        config['vectorSearchConfiguration']['filter'] = filter_obj

    res = agent_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={'text': query},
        retrievalConfiguration=config
    )
    docs = res.get('retrievalResults', [])
    for d in docs:
        logger.info(f"Retrieved doc from KB {kb_id} with metadata={d.get('metadata')}")
    return docs

def merge_results(a, b):
    all_docs = a + b
    all_docs.sort(key=lambda r: r.get('score', 0), reverse=True)
    merged_text = []
    for r in all_docs:
        meta = r.get('metadata', {})
        text = r['content']['text']
        merged_text.append(f"[{meta}] {text}")
    return "\n\n".join(merged_text)

def trim_context(text: str, max_tokens=MAX_CONTEXT_TOKENS):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[-max_tokens:])
    return text

# ----------------- Async LLM Call -----------------
async def call_claude_async(query: str, context_text: str):
    loop = asyncio.get_event_loop()
    prompt = f"""Act strictly as a U.S. healthcare clinical documentation improvement (CDI) analyst. 
You must only answer questions related to U.S. healthcare, medical records, billing, RCM, or patient data. 
Do NOT answer questions about sports, celebrities, politics, or any topic outside U.S. healthcare. 
If the query is unrelated, respond with: "Sorry, I can only answer questions about U.S. healthcare."

Use the following documents to provide your answer:

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
                "max_tokens": 12750,
                "temperature": 0.2
            })
        )
        out = json.loads(response['body'].read().decode())
        return out['content'][0]['text']

    return await loop.run_in_executor(None, invoke)

# ----------------- Async API Route -----------------
@app.route("/chat", methods=["POST"])
async def chat():
    body = request.json
    query = body.get("query")
    session_id = body.get("session_id", "default")
    patient_ids = body.get("patient_ids", [])

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Extract entities
    new_entities = extract_entities(query)

    # Parallel KB retrieval
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        res1_future = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_1, query, patient_ids)
        res2_future = loop.run_in_executor(executor, retrieve_from_kb, KB_ID_2, query, patient_ids)
        res1, res2 = await asyncio.gather(res1_future, res2_future)

    # Update memory
    all_entities = enrich_with_memory(session_id, new_entities, res1 + res2)

    # Merge KB results
    context_text = merge_results(res1, res2)

    # Build full context (cached KB + current + previous) and trim if too long
    memory = session_memory.get(session_id, {})
    cached_docs = memory.get("knowledge_cache", {})
    cached_text = "\n\n".join(cached_docs.values())
    previous_context = memory.get("last_context", "")
    full_context = "\n\n".join(filter(None, [cached_text, context_text, previous_context]))
    full_context = trim_context(full_context)

    # Call Claude
    answer = await call_claude_async(query, full_context)

    # Save trimmed response context
    session_memory[session_id]["last_context"] = (full_context + "\n" + answer).strip()

    logger.info(
        f"Response source=knowledge-base, query='{query[:40]}', entities={all_entities}, patient_ids={patient_ids}"
    )

    return jsonify({
        "answer": answer,
        "source": "knowledge-base",
        "entities": all_entities,
        "patient_ids": patient_ids,
        "session_context": session_memory[session_id]
    })

# ----------------- Frontend Route -----------------
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
