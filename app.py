import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

FILE_PATH = "knowledge_base.txt"

# -----------------------------
# Load Knowledge Base (colon format)
# -----------------------------
def load_kb():
    if not os.path.exists(FILE_PATH):
        st.error(f"❌ File not found: {FILE_PATH}")
        st.stop()

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    faq = []
    for line in lines:
        if ":" in line:
            # Split only at the first colon
            q, a = line.split(":", 1)
            faq.append({
                "question": q.strip(),
                "answer": a.strip()
            })

    if len(faq) == 0:
        st.error("❌ No valid entries found in knowledge_base.txt")
        st.stop()

    return faq


# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Build FAISS Index
# -----------------------------
def build_index(faq, model):
    questions = [item["question"] for item in faq]
    embeddings = model.encode(questions).astype("float32")

    # Safety check
    if embeddings.shape[0] == 0:
        st.error("❌ Error: Empty embedding list. Your knowledge base may be empty.")
        st.stop()

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, questions



def get_answer(query, faq, model, index):
    q_embed = model.encode([query]).astype("float32")
    distances, indices = index.search(q_embed, 1)
    return faq[indices[0][0]]["answer"]



st.title("📘 Knowledge Base Agent (RAG)")

st.write("Ask any HR-related question from your knowledge base!")

faq_data = load_kb()
model = load_model()
index, questions = build_index(faq_data, model)

query = st.text_input("Enter your question:")

if query:
    answer = get_answer(query, faq_data, model, index)
    st.subheader("Answer:")
    st.write(answer)
