# Project Chatbot RAG Blockchain [Hans Joseph B. W. Silitonga]

RAG Chatbot — NoLimit Indonesia Data Scientist Intern Technical Test (Task Option C).

An AI chatbot that answers questions about Bitcoin, Ethereum, and Solana based strictly on their official whitepapers, with cited sources on every answer.

🚀 **Live Demo:** [https://blockchain-rag-chatbot.streamlit.app/](https://blockchain-rag-chatbot.streamlit.app/)

---

## Pipeline

```text
PDF Documents → Text Extraction (PyMuPDF) → Chunking (1000 chars, 150 overlap)
     → Embedding (MiniLM-L6-v2) → FAISS Vector Index
                                          ↓
User Query → Query Embedding → Similarity Search (Top-4 chunks) → RAG Prompt
     → LLM (Qwen2.5-1.5B-Instruct) → Answer + Citation```

See `flowchart.png` for the full visual pipeline diagram.

---

## Dataset / Documents

| Document | Source | License |
|---|---|---|
| Bitcoin Whitepaper | https://bitcoin.org/bitcoin.pdf | Public Domain (MIT) |
| Ethereum Whitepaper | https://ethereum.org/en/whitepaper | CC BY 4.0 |
| Solana Whitepaper | https://solana.com/solana-whitepaper.pdf | Public |

---

## Models

| Role | Model | Source |
|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face |
| Vector Search | `faiss-cpu` (IndexFlatL2) | Meta / PyPI |

---

## Project Structure

```
├── app.py                          # Streamlit web app (bonus)
├── nolimit_ds_rag_chatbot.ipynb    # Notebook (Google Colab)
├── requirements.txt
├── flowchart.png
├── bitcoin.pdf
├── ethereum.pdf
└── solana.pdf
```

---

## Setup & Run

### Requirements
- Python 3.10+
- ~4 GB RAM minimum (8 GB recommended)
- GPU optional but recommended for faster inference

### Installation

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

Make sure `bitcoin.pdf`, `ethereum.pdf`, and `solana.pdf` are in the same folder as `app.py`. On first launch, the app downloads the embedding model and LLM (~3 GB total). Subsequent launches use the cached models.

### Run Notebook

Open `nolimit_ds_rag_chatbot.ipynb` in Google Colab with runtime set to **GPU (T4)**. Follow the steps in order — Step 2 will prompt you to upload the three PDF files.

---

## Example Output

```
Q: What problem does Bitcoin solve?

A: Bitcoin addresses the double-spending problem in peer-to-peer electronic
   transactions without requiring a trusted third party...

Sumber: bitcoin.pdf, Page 1
```
