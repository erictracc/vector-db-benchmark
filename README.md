<h1 align="center">Vector Database Benchmarking: A Survey and Experimental Study</h1>

<p align="center">
  <strong>Author:</strong> Eric Traccitto<br>
  Master’s Student, Information Technology<br>
  York University, Canada 🇨🇦<br>
  <em>erictrac@my.yorku.ca</em>
</p>

---

## Overview

This project presents a **comprehensive benchmarking survey and experimental study** of modern **vector database systems** for semantic search and retrieval tasks.  
The objective is to evaluate the **efficiency**, **scalability**, and **practicality** of various vector search implementations across both open-source and traditional database environments.

The study integrates **five database systems** and measures their performance using a unified benchmarking framework built in Python.  
Each system is tested using the same dataset of AI-related news articles, embedded using OpenAI’s `text-embedding-3-small` model.

---

## Systems Evaluated

Five vector database systems were benchmarked:

| System | Index Type | Vector Engine | Setup |
|:--------|:------------|:---------------|:-------|
| **PostgreSQL + pgvector** | IVF Flat | Native Extension | Server-side |
| **MongoDB + FAISS** | Flat L2 | Client-side (FAISS) | Hybrid |
| **Qdrant** | HNSW | Native Vector Engine | Server-side |
| **Milvus** | IVF Flat | Native Vector Engine | Server-side |
| **Weaviate** | HNSW | Native Vector Engine | Server-side |

Each system was evaluated on:

- 🔹 Search Latency (ms/query)  
- 🔹 Recall Accuracy  
- 🔹 Index Build Time  
- 🔹 Memory Consumption  
- 🔹 Ease of Integration  

---

## Dataset

The dataset consists of **AI-related articles** web-scraped from the **MIT AI News portal**, processed and embedded for semantic comparison.

| File | Description |
|------|--------------|
| `mit_ai_news.csv` | Raw scraped articles |
| `mit_ai_news_embeddings.csv` | Base embeddings using OpenAI API |
| `mit_ai_news_embeddings_expanded.csv` | Extended dataset for additional experiments |

---

## Repository Structure
```
├── .venv/ # Virtual environment
├── data/ # Local data and results
├── models/ # Model storage (if applicable)
├── webscrapped_dataset/
│ ├── mit_ai_news.csv
│ ├── mit_ai_news_embeddings.csv
│ ├── mit_ai_news_embeddings_expanded.csv
│ ├── mongodb_connection.ipynb # MongoDB + FAISS benchmark
│ ├── postgresql_connection.ipynb # PostgreSQL + pgvector benchmark
│ ├── extra_milvus_&_weaviate.ipynb # Milvus & Weaviate benchmarks
│ ├── .env # Environment variables
│ ├── requirements.txt # Dependency list
│ └── ...
├── Real Testing.ipynb # Centralized benchmark and comparison logic
├── slideshow graphs.ipynb # Performance visualization and plots
├── vectordb_connection.ipynb # General connection utilities
├── README.md
└── requirements.txt
```
---

## Methodology

### Embedding Generation
- OpenAI’s `text-embedding-3-small` model was used to generate **1,536-dimensional embeddings**.  
- Data was preprocessed for **uniform vector representation**.

### Database Integration
- Each database was independently configured with the same dataset and index parameters.  
- Both **client-side (FAISS)** and **server-side (pgvector, Milvus, Weaviate, Qdrant)** systems were benchmarked under identical conditions.

### Benchmarking Metrics
-  Query latency (ms/query)  
-  Recall@K accuracy
-  Throughput
-  Resource utilization (memory and CPU)  
-  Ease of integration and deployment  

### Visualization & Analysis
- Results were plotted using **Matplotlib** and **Pandas**.  
- Comparative insights were drawn on performance trade-offs between architectures and index types.

---

## Installation

### 1️Clone the Repository
```
git clone https://github.com/erictraccitto/vector-db-benchmarking.git
cd vector-db-benchmarking
```
2️ Create a Virtual Environment
```
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows
```
3️ Install Dependencies
```
pip install -r requirements.txt
```
(Optional) Freeze Exact Versions
```
pip freeze > requirements.txt
```

Usage
First use the docker config files in order to recreate the containers which create the database management system:
- docker-compose.yml
- docker-compose-extended.yml

Run the benchmarking notebooks sequentially to reproduce results:

Real Testing.ipynb

extra_milvus_&_weaviate.ipynb

Research Contribution
This study provides a comparative foundation for understanding how vector database architectures impact performance in AI-driven retrieval systems.

The findings can guide:

Database selection for LLM-powered applications

Optimization of semantic search pipelines

Evaluation of indexing strategies (IVF vs HNSW)

Future Directions
Expand dataset size for scalability testing

Integrate distributed environments (e.g., Milvus cluster mode)

Compare cloud-hosted systems (e.g., Pinecone, ChromaDB)

Explore integration with ChatGPT’s retrieval-augmented generation (RAG) workflows

<p align="center"> <sub>© 2025 Eric Traccitto — York University<br> Licensed under the <a href="https://opensource.org/licenses/MIT">MIT License</a>.</sub> </p>


