# 🚀 Knowledge Base Agent (RAG)

A lightweight, fast, and fully deployable HR Knowledge Base Assistant built using Streamlit, Sentence Transformers, and FAISS.

---

## 📘 Overview

The Knowledge Base Agent (RAG) is an intelligent HR FAQ assistant that allows users to ask questions in natural language and retrieves the most relevant answer from a local knowledge base.

It uses **Retrieval-Augmented Generation (RAG)** architecture with:  

- **Sentence Transformers** for semantic embeddings  
- **FAISS** for fast similarity search  
- **Streamlit** for the web interface  

No external API keys, no OpenAI cost — everything runs locally or on Streamlit Cloud.

---

## ⭐ Features

- **🔍 Semantic Search (RAG)**: Answers HR-related questions based on your knowledge base and retrieves the closest matching entry using embeddings.  
- **🧠 Lightweight On-Device Model**: Uses `all-MiniLM-L6-v2`, fast and accurate; no GPU required.  
- **📁 Flexible Knowledge Base Format**: Data stored in `key: value` format; the agent automatically parses the file.  
- **⚡ Fast & Efficient**: Rapid inference, minimal compute usage, fully deployable on Streamlit Cloud.  
- **🎨 User-Friendly Interface**: Clean input box, simple output view, works on mobile & desktop.

---

## 🧠 How It Works

1. Loads `knowledge_base.txt`  
2. Splits each line into `question → answer`  
3. Embeds questions using **SentenceTransformer**  
4. Builds a **FAISS vector index**  
5. Takes user query → converts to embedding  
6. FAISS finds top matching question  
7. Corresponding answer is displayed  

This is the core of **RAG**, without the generation part.

---

## 🛠️ Tech Stack

| Component | Purpose |
|-----------|---------|
| Streamlit | UI & deployment |
| Sentence-Transformers | Embeddings |
| FAISS CPU | Vector similarity search |
| Torch | Backend for embeddings |
| NumPy | Numerical operations |
