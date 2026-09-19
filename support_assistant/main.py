import os, json, re
from typing import TypedDict, Literal, List
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "zepto_policies"

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

class AskRequest(BaseModel):
    query: str

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    context: List[str]
    source_ids: List[str]
    answer: AnswerResponse

embedder = SentenceTransformer(MODEL_NAME)
chroma = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def ingest():
    """Load each file as one chunk and index it."""
    docs = []
    ids = []
    metadatas = []
    for filename in sorted(os.listdir("docs")):
        if filename.endswith(".txt"):
            text = open(os.path.join("docs", filename), encoding="utf-8").read().strip()
            doc_id = filename.rsplit(".", 1)[0]
            docs.append(text)
            ids.append(doc_id)
            metadatas.append({"document_id": doc_id, "filename": filename})
    if docs:
        embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    if MOCK_LLM:
        keywords = ["delivery", "return", "refund", "membership", "tracking",
                    "cancel", "gift card", "support hours"]
        intent = "policy_question" if any(k in query.lower() for k in keywords) else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    q_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=q_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    documents = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []
    source_ids = [m["document_id"] for m in metas]
    if not documents:
        return {"context": [], "source_ids": [], "answer": AnswerResponse(
            answer="No relevant policy context was retrieved.",
            sources=[], confidence=0.0
        )}

    if MOCK_LLM:
        snippet = documents[0][:200]
        answer = f"Based on the retrieved context: {snippet}"
        return {"context": documents, "source_ids": source_ids,
                "answer": AnswerResponse(answer=answer, sources=source_ids, confidence=1.0)}

def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
        return {"answer": AnswerResponse(answer=answer, sources=[], confidence=1.0)}

def route(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

PROMPT_TEMPLATE = """ROLE:
You are a Zepto support assistant.

CONTEXT:
Use only the Zepto policy context supplied below.
{context}

TASK:
Answer the customer's question using the supplied context.

FORMAT:
Return JSON with exactly these fields:
{{"answer": "string", "sources": ["document ids"], "confidence": 0.0}}

LENGTH:
Keep the answer concise and directly useful.

NEGATIVE CONSTRAINT:
Do not answer using information not present in the provided context. Do not invent or assume policy details.

FEW-SHOT EXAMPLE:
Question: "What is the delivery fee below INR 149?"
Context: "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
Output: {{"answer":"Orders below INR 149 incur a flat INR 25 delivery fee.","sources":["doc_01"],"confidence":1.0}}
"""


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()

ingest()
app = FastAPI(title="Zepto Support Assistant")
graph = build_graph()

@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})
    return result["answer"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
