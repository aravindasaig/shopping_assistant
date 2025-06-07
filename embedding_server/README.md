

# 🧠 Embedding Generation Service

This service provides APIs for generating **text** and **image embeddings** using the `llamaindex/vdr-2b-multi-v1` model. It supports processing of:

* Raw text input
* Images (file upload or base64)
* Combined structured content (with metadata) for full document processing

## 🛠️ Tech Stack

* **FastAPI** – Web API Framework
* **llamaindex** – Embedding model backend
* **HuggingFace Embeddings** – Text + Image
* **Pillow** – Image manipulation
* **NumPy** – Vector ops

## ⚙️ Configuration
Embedding model used
* `llamaindex/vdr-2b-multi-v1`
* Use `trust_remote_code=True` if needed

---

## 🧪 Run the Service

```bash
/bin/nohup python3 embedding_server.py > results.log&
```

---


### 🔹 `/embed-image` (POST)

Generate an embedding for a single uploaded image file.

#### Form Data

* `file`: Image file (`.png`, `.jpg`, `.jpeg`, etc.)

#### Response

```json
{
  "embedding": [...],
  "dimensions": 1024,
  "filename": "example.png"
}
```

---

### 🔹 `/generate-text-embedding` (POST)

Generate an embedding for a single text string.

#### Request Body

```json
{
  "text": "Artificial Intelligence is transforming industries."
}
```

#### Response

```json
{
  "embedding": [...],
  "dimensions": 1024,
  "text_length": 48
}
```

---

### 🔹 `/health` (GET)

Health check for the service.

#### Response

```json
{
  "status": "ok",
  "service": "embedding_generation"
}
```

---

### 🔹 `/debug/info` (GET)

Debug endpoint to inspect internal state.

---

## 📁 Project Structure

```
.
├── embedding_generator.py    # Core logic for embedding generation
├── embedding_api.py          # FastAPI web server with routes
├── logger_config.py          # Centralized logger setup
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 🧼 Temp File Handling

* All image files are stored temporarily in the OS tmp dir.
* Cleaned up after processing automatically.

---

## 🧑‍💻 Author

Aravinda Sai Gadamsetty

---

