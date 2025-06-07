# RAG API Service

A unified API service for vector database interaction (Weaviate & Milvus) to support retrieval-augmented generation (RAG).

---

## 🚀 Key Features

* 🔍 **Semantic Search** via Azure OpenAI embeddings
* 🗂️ **Dynamic Collection Creation** with custom schema and multiple vectors
* 📥 **Flexible Data Insertion** with nested properties
* 🔧 **Configurable** via `config.json`
* 🧠 **Supports** both **Weaviate** and **Milvus**

---

## 📁 Project Structure


rag-api-service/
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
└── src/              # Source code modules
    ├── config.json   # Configuration file
    ├── config.py     # Configuration loader module
    ├── main.py       # Main application entry point
    ├── utils.py
    ├── weaviate_query.py
    ├── weaviate_read.py
    ├── weaviate_update.py
    └── logging_config.py


---

## 🔧 Running Locally

bash
pip install -r requirements.txt
python -m src.main


---

## 📘 Sample `config.json`

json
{
  "azure_openai": {
    "api_key": "...",
    "endpoint": "...",
    "api_version": "2023-05-15"
  },
  "vector_db": {
    "type": "weaviate",
    "weaviate": {
      "http_host": "localhost",
      "http_port": 8080
    }
  },
  "service": {
    "port": 8000,
    "host": "0.0.0.0"
  }
}


---

## 📌 API Usage

### ✅ Create Collection

bash
curl -X POST http://localhost:8000/create \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "articles",
    "properties": [
      { "name": "title", "data_type": "text" },
      { "name": "content", "data_type": "text" }
    ],
    "vectors": [
      { "name": "content_vector", "vector_type": "none", "dimensions": 1536 }
    ]
  }'


---

### 📥 Insert Data

bash
curl -X POST http://localhost:8000/insert \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "articles",
    "data": [
      {
        "properties": {
          "title": "Intro to RAG",
          "content": "RAG combines retrieval and generation."
        },
        "vectors": {
          "content_vector": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
      }
    ]
  }'


---

### 🔍 Search

bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "articles",
    "query": {
      "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    "columns": {
      "content_vector": 1.0
    },
    "output_fields": ["title", "content"],
    "top_k": 1
  }'


---

## 📎 Health Check

bash
curl http://localhost:8000/status



