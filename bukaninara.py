import streamlit as st
import os
from openai import OpenAI
from pinecone import Pinecone, Index
import time

# --- Konfigurasi Awal ---
st.set_page_config(page_title="Chatbot Tanya Jawab dengan Pinecone & OpenAI", layout="wide")

# --- Fungsi Utility ---

@st.cache_resource
def init_pinecone(api_key, environment):
    """Menginisialisasi Pinecone."""
    try:
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        st.error(f"Gagal menginisialisasi Pinecone: {e}")
        return None

@st.cache_resource
def get_embedding_model(api_key):
    """Menginisialisasi klien OpenAI untuk embedding."""
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Gagal menginisialisasi OpenAI untuk embedding: {e}")
        return None

def get_embedding(text, client):
    """Mendapatkan embedding untuk teks."""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Gagal mendapatkan embedding: {e}")
        return None

def upsert_to_pinecone(index_name, texts, metadata_list, pinecone_client, embedding_client):
    """Mengunggah data ke Pinecone."""
    if not pinecone_client:
        st.error("Pinecone belum diinisialisasi. Tidak dapat mengunggah.")
        return False
    
    if index_name not in pinecone_client.list_indexes().names():
        st.info(f"Membuat indeks Pinecone '{index_name}'...")
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,  # Sesuai dengan text-embedding-ada-002
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-west-2"}} # Contoh, sesuaikan dengan region Anda
        )
        # Tunggu sampai indeks siap
        while not pinecone_client.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pinecone_client.Index(index_name)
    
    vectors_to_upsert = []
    for i, text in enumerate(texts):
        embedding = get_embedding(text, embedding_client)
        if embedding:
            vectors_to_upsert.append({
                "id": f"doc-{time.time()}-{i}", # ID unik
                "values": embedding,
                "metadata": metadata_list[i]
            })
    
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            st.success(f"Berhasil mengunggah {len(vectors_to_upsert)} vektor ke Pinecone indeks '{index_name}'.")
            return True
        except Exception as e:
            st.error(f"Gagal mengunggah ke Pinecone: {e}")
            return False
    return False

def query_pinecone(index_name, query_text, top_k, pinecone_client, embedding_client):
    """Mencari di Pinecone."""
    if not pinecone_client:
        st.error("Pinecone belum diinisialisasi. Tidak dapat melakukan pencarian.")
        return []
    
    if index_name not in pinecone_client.list_indexes().names():
        st.warning(f"Indeks '{index_name}' tidak ditemukan.")
        return []

    index = pinecone_client.Index(index_name)
    query_embedding = get_embedding(query_text, embedding_client)
    if query_embedding:
        try:
            res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            return res.matches
        except Exception as e:
            st.error(f"Gagal melakukan query Pinecone: {e}")
            return []
    return []

def get_answer_from_openai(question, context, openai_api_key):
    """Mendapatkan jawaban dari OpenAI GPT."""
    try:
        client = OpenAI(api_key=openai_api_key)
        messages = [
            {"role": "system", "content": "Anda adalah asisten AI yang cerdas. Jawab pertanyaan berdasarkan konteks yang diberikan."},
            {"role": "user", "content": f"Konteks: {context}\n\nPertanyaan: {question}"}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Model yang disarankan
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Gagal mendapatkan jawaban dari OpenAI: {e}")
        return "Maaf, saya tidak bisa menjawab pertanyaan ini saat ini."

# --- Sidebar untuk Konfigurasi ---
st.sidebar.title("Konfigurasi API Key")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
pinecone_environment = st.sidebar.text_input("Pinecone Environment (e.g., us-west-2-aws)", "us-west-2-aws") # Contoh, sesuaikan

# --- Inisialisasi Klien ---
pinecone_client = None
openai_embedding_client = None

if openai_api_key:
    openai_embedding_client = get_embedding_model(openai_api_key)
else:
    st.sidebar.warning("Masukkan OpenAI API Key Anda.")

if pinecone_api_key and pinecone_environment:
    pinecone_client = init_pinecone(pinecone_api_key, pinecone_environment)
else:
    st.sidebar.warning("Masukkan Pinecone API Key dan Environment Anda.")

# --- Bagian Upload Database ---
st.header("Unggah Database ke Pinecone")
st.write("Unggah file teks Anda untuk dijadikan database chatbot.")

uploaded_file = st.file_uploader("Pilih file teks (.txt)", type=["txt"])
index_name = st.text_input("Nama Indeks Pinecone (contoh: my-chatbot-data)", "my-chatbot-data")

if uploaded_file and index_name:
    if st.button("Proses dan Unggah ke Pinecone"):
        if openai_embedding_client and pinecone_client:
            with st.spinner("Memproses dan mengunggah database..."):
                file_content = uploaded_file.read().decode("utf-8")
                
                # Sederhana: memisahkan berdasarkan baris baru untuk contoh ini
                # Untuk kasus nyata, Anda mungkin perlu chunking yang lebih canggih
                texts = [line.strip() for line in file_content.split('\n') if line.strip()]
                metadata_list = [{"source": uploaded_file.name, "line": i} for i, _ in enumerate(texts)]

                if texts:
                    upsert_to_pinecone(index_name, texts, metadata_list, pinecone_client, openai_embedding_client)
                else:
                    st.warning("File tidak berisi teks yang dapat diproses.")
        else:
            st.error("Harap masukkan API Key yang valid.")

st.markdown("---")

# --- Bagian Chatbot Tanya Jawab ---
st.header("Chatbot Tanya Jawab")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan pesan chat sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan sesuatu..."):
    # Tambahkan pesan pengguna ke chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if openai_api_key and pinecone_api_key and pinecone_environment:
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                # 1. Query Pinecone untuk konteks
                pinecone_results = query_pinecone(index_name, prompt, top_k=3, pinecone_client=pinecone_client, embedding_client=openai_embedding_client)
                
                context = ""
                if pinecone_results:
                    context_texts = [match.metadata['text'] if 'text' in match.metadata else ' '.join(match.id.split('-')) for match in pinecone_results] # Mengambil teks dari metadata
                    # Jika metadata tidak memiliki 'text', coba ambil dari ID atau fallback
                    
                    # Coba ambil konten sebenarnya dari file_content jika ada
                    # Untuk implementasi yang lebih robust, Anda perlu menyimpan teks asli saat upsert
                    # Contoh sederhana:
                    if uploaded_file and 'text_cache' in st.session_state and uploaded_file.name in st.session_state.text_cache:
                         all_lines = st.session_state.text_cache[uploaded_file.name].split('\n')
                         context_texts = []
                         for match in pinecone_results:
                             try:
                                 line_num = match.metadata.get('line')
                                 if line_num is not None and line_num < len(all_lines):
                                     context_texts.append(all_lines[line_num])
                                 elif 'text' in match.metadata: # Fallback jika tidak ada line_num
                                     context_texts.append(match.metadata['text'])
                             except Exception as e:
                                 st.warning(f"Gagal mengambil konteks dari baris: {e}")
                    elif pinecone_results:
                         context_texts = [match.metadata.get('text', match.id) for match in pinecone_results] # Fallback umum
                        
                    context = "\n".join(context_texts)
                    
                if not context:
                    st.warning("Tidak menemukan konteks yang relevan di database Anda. Akan mencoba menjawab dengan pengetahuan umum OpenAI.")
                    
                # 2. Dapatkan jawaban dari OpenAI
                answer = get_answer_from_openai(prompt, context, openai_api_key)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Harap masukkan semua API Key dan Konfigurasi Pinecone di sidebar.")

st.markdown("---")
st.caption("Dibuat dengan ❤️ oleh AI")

# --- Untuk menyimpan teks yang diunggah sementara (untuk konteks) ---
if uploaded_file:
    if 'text_cache' not in st.session_state:
        st.session_state.text_cache = {}
    st.session_state.text_cache[uploaded_file.name] = uploaded_file.read().decode("utf-8")
