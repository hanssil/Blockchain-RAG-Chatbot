import streamlit as st
import fitz
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Blockchain RAG Chatbot", page_icon="🤖")
st.title("Blockchain RAG Chatbot 🤖")
st.caption("Asisten AI berbasis dokumen whitepaper Bitcoin, Ethereum, dan Solana.")

# ==========================================
# 1. CACHING: Load AI Models (once per session)
# ==========================================
@st.cache_resource(show_spinner="Memuat Model AI... (butuh beberapa menit di awal)")
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )
    return embedder, llm, tokenizer

embedder, llm, tokenizer = load_models()


# ==========================================
# 2. CACHING: Build FAISS Vector DB (once per session)
# ==========================================
@st.cache_resource(show_spinner="Membaca PDF dan membangun Vector Database...")
def build_vector_db():
    pdf_files  = ["bitcoin.pdf", "ethereum.pdf", "solana.pdf"]
    all_chunks = []

    for filename in pdf_files:
        if not os.path.exists(filename):
            st.error(f"File tidak ditemukan: {filename}")
            continue

        doc = fitz.open(filename)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if not text:
                continue
            # Chunking with overlap
            start = 0
            while start < len(text):
                chunk = text[start : start + 1000]
                all_chunks.append({
                    "text":        chunk,
                    "file_name":   filename,
                    "page_number": page_num + 1
                })
                start += 1000 - 150
        doc.close()

    # Build FAISS index
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings  = embedder.encode(chunk_texts, convert_to_numpy=True)
    index       = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    return index, all_chunks

index, all_chunks = build_vector_db()


# ==========================================
# 3. RAG HELPER FUNCTIONS
# ==========================================
def retrieve(query, top_k=4):
    query_vec  = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(query_vec, top_k)
    return [all_chunks[i] for i in indices[0]]


def format_citations(chunks):
    """Deduplicated citation string, preserving retrieval order."""
    seen, parts = set(), []
    for c in chunks:
        key = (c["file_name"], c["page_number"])
        if key not in seen:
            seen.add(key)
            parts.append(f"{c['file_name']}, Page {c['page_number']}")
    return " | ".join(parts)


def generate_answer(user_question, retrieved_chunks):
    context_parts = [
        f"[Passage {i}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, 1)
    ]
    context = "\n\n".join(context_parts)

    system_msg = (
        "You are an AI assistant. Answer the user's question using ONLY the provided context.\n"
        "1. Reply in the EXACT SAME LANGUAGE as the user's prompt. Keep technical terms as-is.\n"
        "2. If the answer is not in the context, or the input is a random word, do NOT guess and do NOT echo the input.\n"
        "3. If you cannot answer based on the context, you MUST start your response with the exact tag '[NOT_FOUND]' followed by an apology in the user's language (e.g., '[NOT_FOUND] Maaf, informasi tidak ditemukan.' or '[NOT_FOUND] Sorry, the information is not found.')."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {user_question}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm(prompt)[0]["generated_text"]
    return output[len(prompt):].strip()  # strip prompt prefix, return only new text


# ==========================================
# 4. CHAT UI
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.info("💡 Coba tanyakan: *'What is proof of work?'* atau *'Bagaimana Solana mencapai kecepatan tinggi?'*")

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt_user := st.chat_input("Tanyakan sesuatu tentang Bitcoin, Ethereum, atau Solana..."):

    with st.chat_message("user"):
        st.markdown(prompt_user)
    st.session_state.messages.append({"role": "user", "content": prompt_user})

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban di dokumen..."):
            chunks         = retrieve(prompt_user)
            answer         = generate_answer(prompt_user, chunks)
            citation_text  = format_citations(chunks)

            # --- LOGIKA FILTERING MULTI-BAHASA ---
            if "[NOT_FOUND]" in answer:
                clean_answer = answer.replace("[NOT_FOUND]", "").strip()
                final_response = clean_answer  # Tampilkan jawaban bersih tanpa sitasi
            else:
                final_response = f"{answer}\n\n**Sumber:** {citation_text}"

        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
