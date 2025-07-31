---

## **Complete README.md for All 4 Projects**

````markdown
# GenAI Project Showcase 🚀

This repository contains multiple small projects that demonstrate the use of Generative AI techniques like Vector Databases, Retrieval-Augmented Generation (RAG), Custom Model Development, and Fine-tuning of Pretrained Models. The implementations are done using Python, PyTorch, ChromaDB, and HuggingFace Transformers.

---

## 📂 Project Overview

### 1. Vector Name Matching 🔍
**Folder:** `VectorNameMatch/`

- Objective: Match user-input names with the most similar names from a database.
- Techniques Used: Vector Embeddings, ChromaDB Vector Search.
- Features:
  - Load multiple similar names into a Vector DB.
  - Retrieve the closest matching name(s) with similarity scores.
- Tech Stack: Python, Sentence Transformers, ChromaDB.

---

### 2. Recipe Chatbot using LLM APIs 🍽️
**Folder:** `Recipe Chatbot/`

- Objective: Build a chatbot that answers user queries about recipes.
- Techniques Used: LLM API Integration, RAG Architecture.
- Features:
  - API-based interaction with LLMs for dynamic responses.
  - Outputs structured JSON with recipe suggestions.
- Tech Stack: Python, PyTorch, API Calls, RAG Flow.

---

### 3. Custom Q&A Model (Built from Scratch) 🤖
**Folder:** `own_model/`

- Objective: Develop a custom model from scratch to handle domain-specific Q&A.
- Techniques Used: PyTorch Model Development & Training.
- Features:
  - Custom Neural Network architecture.
  - Trained using domain-specific data for Q&A.
  - Inference-ready scripts for integration.
- Tech Stack: Python, PyTorch.

---

### 4. Fine-Tuned T5 Small Model (LoRA + PEFT) 🧠
**Folder:** `own_model_with_t5_small_pretrained_model/`

- Objective: Fine-tune a pretrained T5-small model for Q&A tasks using LoRA and PEFT.
- Techniques Used: HuggingFace Transformers, LoRA (Low-Rank Adaptation), PEFT.
- Features:
  - Lightweight fine-tuning to adapt T5-small to a specific domain.
  - Efficient training with minimal resources.
  - Inference-ready fine-tuned model.
- Tech Stack: Python, HuggingFace, PyTorch, PEFT.

---

## 🛠️ Tech Stack Summary
- Python
- PyTorch
- ChromaDB (Vector Database)
- HuggingFace Transformers
- LoRA & PEFT Fine-Tuning
- API Integrations
- FastAPI (For Local APIs)
- Docker (Optional for Deployment)

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/<your-username>/GenAI-Project-Showcase.git
cd GenAI-Project-Showcase

# Navigate to specific project folders and follow their individual instructions.
````

---

## Folder Structure

```
GenAI-Project-Showcase/
├── own_model/                          # Custom Model from Scratch (PyTorch)
├── own_model_with_t5_small_pretrained_model/  # Fine-tuned T5 Small Model (LoRA + PEFT)
├── Recipe Chatbot/                     # Recipe Chatbot using LLM API
├── VectorNameMatch/                    # Name Matching using ChromaDB Vector Search
└── README.md                           # (This File)
```

---

## ✍️ Author

**Awanish Kumar**
Generative AI Engineer | Python | LLMs | Vector DB | PyTorch

---
