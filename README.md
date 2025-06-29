# Multimodal RAG PDF Assistant

**Multimodal Retrieval-Augmented Generation (RAG) pipeline that answers questions about PDF documents combining advanced vision & language models.**

---

## ✨ Key Components

| Stage              | Model                              | HF Repo                                                                             |
| ------------------ | ---------------------------------- | ----------------------------------------------------------------------------------- |
| Document retrieval | **ColQwen2**                       | [`vidore/colqwen2-v1.0-merged`](https://huggingface.co/vidore/colqwen2-v1.0-merged) |
| Re-ranking         | **MonoVLM** (cross-modal reranker) | via [`rerankers[monovlm]`](https://github.com/sergiopaniego/rerankers)              |
| Answer generation  | **Qwen2-VL-7B-Instruct**           | [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)     |

1. **Index & retrieve** relevant PDF pages (rendered to images) with ColQwen2.
2. **Rerank** the images against the user query using MonoVLM for better context selection.
3. **Generate** the final natural-language answer with Qwen2-VL-7B.

Everything is wrapped in a **Streamlit** web UI so you can simply upload PDFs and ask questions.

---

## 🚀 Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/sophic00/CEWXAI1IN-project.git
uv add -r requirements.txt
source .venv/bin/activate
```

> On Linux you also need Poppler utilities for PDF → image conversion:
>
> ```bash
> pacman -S poppler
> ```

### 2 — Run the app

```bash
streamlit run src/app.py
```

Open the local URL (defaults to <http://localhost:8501>) and:

1. Upload one or multiple PDF files in the sidebar.
2. Choose retrieval and rerank _top-k_ sliders.
3. Type a question and click **Get Answer**.

The answer plus the exact page images used for reasoning will be displayed.

### 3 — Docker (optional)

A ready-to-use `Dockerfile` is included. Build & run:

```bash
docker build -t multimodal-rag .
docker run -p 8501:8501 --gpus all multimodal-rag
```

> GPU access is strongly recommended (16 GB VRAM ≈ L4 / RTX A6000). On CPU the models will work but can be **very** slow.

---

## ⚙️ Project Structure

```
├── src/
│   ├── app.py           # Streamlit UI
│   ├── model_loader.py  # Loads retrieval, reranker, VLM
│   ├── data_processing.py
│   └── rag_pipeline.py  # Core logic
├── requirements.txt
├── Dockerfile
└── pyproject.toml
```

---

## 📝 Configuration

Environment variables (all optional):

| Variable                   | Description                    |
| -------------------------- | ------------------------------ |
| `TRANSFORMERS_CACHE`       | Path to reuse HF model cache   |
| `HF_HUB_DISABLE_TELEMETRY` | Set to `1` for offline privacy |

---

## 📚 References & Inspiration

- Hugging Face Cookbook – [Multimodal RAG with ColQwen2, Reranker, and Quantized VLMs](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_reranker_and_vlms)
