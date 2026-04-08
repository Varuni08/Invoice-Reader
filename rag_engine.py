import hashlib
import re
from typing import Optional
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

MODEL_NAME  = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_NAME  = "invoice-rag"
DIMENSION   = 384
CHUNK_SIZE  = 400
CHUNK_OVERLAP = 80

_embedder: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_index(pinecone_api_key: str):
    pc = Pinecone(api_key=pinecone_api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def index_invoice(pinecone_api_key: str, invoice_id: str, text: str, metadata: dict) -> int:
    index    = get_index(pinecone_api_key)
    embedder = get_embedder()
    chunks   = chunk_text(text)

    vectors = []
    for i, chunk in enumerate(chunks):
        vec_id    = hashlib.md5(f"{invoice_id}_{i}".encode()).hexdigest()
        embedding = embedder.encode(chunk).tolist()
        vectors.append({
            "id":     vec_id,
            "values": embedding,
            "metadata": {
                **metadata,
                "invoice_id": invoice_id,
                "chunk_idx":  i,
                "text":       chunk,
            },
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    return len(vectors)


def delete_invoice(pinecone_api_key: str, invoice_id: str):
    index = get_index(pinecone_api_key)
    index.delete(filter={"invoice_id": {"$eq": invoice_id}})


def retrieve_context(pinecone_api_key: str, invoice_id: str, query: str, top_k: int = 5) -> list:
    index    = get_index(pinecone_api_key)
    embedder = get_embedder()
    q_vec    = embedder.encode(query).tolist()
    results  = index.query(
        vector=q_vec,
        top_k=top_k,
        filter={"invoice_id": {"$eq": invoice_id}},
        include_metadata=True,
    )
    return [m["metadata"]["text"] for m in results.get("matches", [])]


def build_system_prompt(inv: dict) -> str:
    vendor  = inv.get("vendor") or "Unknown vendor"
    inv_num = inv.get("invoiceNumber") or "N/A"
    total   = inv.get("total")
    currency = inv.get("currency") or "USD"
    due     = inv.get("dueDate") or "N/A"
    terms   = inv.get("paymentTerms") or "N/A"

    fin  = inv.get("_financial") or {}
    val  = inv.get("_validation") or {}
    dup  = inv.get("_duplicate") or {}

    flags     = val.get("flags") or []
    urgency   = fin.get("urgencyLabel") or "N/A"
    early_pay = fin.get("earlyPayDiscount") or {}
    is_dup    = dup.get("isDuplicate", False)
    dup_match = dup.get("matchedWith", "")
    round_num = val.get("roundNumber", False)
    back_tax  = val.get("backCalcTaxRate")

    flag_summary = "; ".join(f["msg"] for f in flags) if flags else "None"

    return f"""You are a financial assistant specialising in invoice analysis.
You have full knowledge of the following invoice and must answer questions accurately.

--- INVOICE SUMMARY ---
Vendor:           {vendor}
Invoice #:        {inv_num}
Total:            {currency} {total}
Due date:         {due}
Payment terms:    {terms}
Urgency:          {urgency}
Duplicate flag:   {"YES — matches " + dup_match if is_dup else "No"}
Round number:     {"Yes — verify manually" if round_num else "No"}
Back-calc tax:    {f"{back_tax}%" if back_tax else "N/A"}
Validation flags: {flag_summary}
{"Early pay offer: " + early_pay.get("label","") if early_pay else ""}
-----------------------

Rules:
- Answer concisely and factually based on the invoice data and retrieved context.
- If the answer is not in the data, say so clearly.
- Do not make up numbers.
- When referencing amounts always include the currency symbol.
- Keep responses to 3-5 sentences unless detail is explicitly requested.
"""


def answer_question(groq_client: Groq, pinecone_api_key: str, invoice_id: str, inv: dict, question: str, history: list) -> str:
    context_chunks = retrieve_context(pinecone_api_key, invoice_id, question)
    context_text   = "\n\n".join(context_chunks) if context_chunks else "No additional context found."

    system   = build_system_prompt(inv)
    messages = [{"role": "system", "content": system}]

    for msg in history[-10:]:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"Relevant invoice context:\n{context_text}\n\nQuestion: {question}"
    })

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=600,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def suggest_questions(inv: dict) -> list:
    suggestions = []
    fin = inv.get("_financial") or {}
    val = inv.get("_validation") or {}
    dup = inv.get("_duplicate") or {}

    urgency = fin.get("urgency")
    if urgency in ("overdue", "critical"):
        suggestions.append("This invoice is overdue — can you draft a payment reminder?")
    elif urgency in ("high", "medium"):
        suggestions.append("How many days until this invoice is due?")

    if fin.get("earlyPayDiscount"):
        suggestions.append("How much can I save by paying early?")

    flags = [f for f in (val.get("flags") or []) if f["type"] == "error"]
    if flags:
        suggestions.append("What validation errors were found in this invoice?")

    if dup.get("isDuplicate"):
        suggestions.append("Why is this flagged as a potential duplicate?")

    if val.get("roundNumber"):
        suggestions.append("Why is the round number total flagged?")

    if inv.get("paymentTerms"):
        suggestions.append("Explain the payment terms in plain English.")

    if inv.get("lineItems"):
        suggestions.append("Break down the line items for me.")

    if inv.get("taxAmount"):
        suggestions.append("What is the effective tax rate on this invoice?")

    suggestions.append("Summarise this invoice for me.")

    return suggestions[:5]
