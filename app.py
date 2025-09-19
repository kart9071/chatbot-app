import os, json, logging, sys, re
from flask import Flask, request, jsonify, render_template
import boto3

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# --- Env Vars ---
KB_ID_1 = "WDBONTXTWY"   # Currently not used, left here for future
KB_ID_2 = "TKWQDABBSG"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
AWS_REGION = "us-east-1"

# --- AWS Clients ---
agent_client = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
comprehend = boto3.client("comprehendmedical", region_name=AWS_REGION)

# --- Flask ---
app = Flask(__name__)

# --- Simple in-memory session storage ---
# Stores entities + last_context for follow-up queries
session_memory = {}

# ----------------- Entity Extraction -----------------
def extract_entities(query: str):
    """Use Comprehend Medical for entity extraction."""
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
        # fallback regex
        m = re.search(r"\bmember\s+(\w+)", query, re.I)
        if m:
            entities["member_id"] = m.group(1)
        p = re.search(r"\bpatient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", query)
        if p:
            entities["patient_name"] = p.group(1)

    return entities


def enrich_with_memory(session_id, entities):
    """Keep memory of entities across session turns."""
    if session_id not in session_memory:
        session_memory[session_id] = {}
    memory = session_memory[session_id]
    memory.update(entities)
    for k, v in memory.items():
        if k not in entities:
            entities[k] = v
    return entities

# ----------------- KB Retrieval -----------------
def merge_results(a, b):
    """Merge and sort results from two KBs, return combined text."""
    all_docs = a + b
    all_docs.sort(key=lambda r: r.get('score', 0), reverse=True)

    merged_text = []
    for r in all_docs:
        meta = r.get('metadata', {})
        text = r['content']['text']
        logger.info(f"📄 Retrieved metadata: {meta}")
        merged_text.append(f"[{meta}] {text}")

    return "\n\n".join(merged_text)


def retrieve_from_kb(kb_id: str, query: str, patient_ids=None):
    """Retrieve documents from Bedrock Knowledge Base with optional filtering."""
    filter_obj = None
    if patient_ids:
        if len(patient_ids) == 1:
            filter_obj = {"equals": {"key": "patient_id", "value": patient_ids[0]}}
        elif len(patient_ids) > 1:
            filter_obj = {"in": {"key": "patient_id", "value": patient_ids}}

    config = {'vectorSearchConfiguration': {'numberOfResults': 10}}
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

# ----------------- LLM Call -----------------
def call_claude(query: str, context_text: str):
    """Call Claude model with context + query."""
    prompt = f"""Use the following documents to answer the question accurately.

Context:
{context_text}

Question: {query}
Answer:"""
    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": 5000,
            "temperature": 0.2
        })
    )
    out = json.loads(response['body'].read().decode())
    return out['content'][0]['text']

# ----------------- API Route -----------------
@app.route("/chat", methods=["POST"])
def chat():
    body = request.json
    query = body.get("query")
    session_id = body.get("session_id", "default")
    patient_ids = body.get("patient_ids", [])

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Extract new entities
    new_entities = extract_entities(query)
    all_entities = enrich_with_memory(session_id, new_entities)

    # --- Include prior response context if available ---
    previous_context = session_memory.get(session_id, {}).get("last_context", "")
    if previous_context:
        full_query = query + f"\n[previous context: {previous_context}]"
    else:
        entity_text = ", ".join(f"{k}: {v}" for k, v in all_entities.items())
        full_query = query + (f"\n[context: {entity_text}]" if entity_text else "")

    # --- Retrieval ---
    res1 = retrieve_from_kb(KB_ID_1, full_query, patient_ids)   # Optional KB
    res2 = retrieve_from_kb(KB_ID_2, full_query, patient_ids)

    # Enrich entities from metadata
    for doc in res1 + res2:
        meta = doc.get("metadata", {})
        if "patient_name" in meta:
            all_entities["patient_name"] = meta["patient_name"]
        if "patient_id" in meta:
            if "patient_ids" not in all_entities:
                all_entities["patient_ids"] = []
            if meta["patient_id"] not in all_entities["patient_ids"]:
                all_entities["patient_ids"].append(meta["patient_id"])

    # Merge context
    context_text = merge_results(res1, res2)
    source = "knowledge-base"

    # Call Claude
    answer = call_claude(full_query, context_text)

    # --- Save response context for follow-up queries ---
    session_memory[session_id]["last_context"] = answer

    logger.info(
        f"Response source={source}, query='{query[:40]}' entities={all_entities} patient_ids={patient_ids}"
    )
    return jsonify({
        "answer": answer,
        "source": source,
        "entities": all_entities,
        "patient_ids": patient_ids,
        "session_context": session_memory[session_id]  # helpful for debugging
    })

# ----------------- Frontend Route -----------------
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
