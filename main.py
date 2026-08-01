import os
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

# 1. Doküman Yükleme ve Parçalama
def load_and_split_docs(file_path):
    print(f"\n[1/4] Doküman yükleniyor: {file_path}")
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = text_splitter.split_documents(documents)
    print(f"[✓] Doküman {len(chunks)} parçaya bölündü.")
    return chunks

# 2. Vektör Veritabanı Oluşturma
def create_vector_db(chunks):
    print("[2/4] Vektör veritabanı hazırlanıyor...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("[✓] Vektör veritabanı hazır.")
    return vector_db

# 3. Yerel Python Modeli ile Cevap Üretme
def generate_answer(vector_db, query):
    print(f"\n[3/4] Vektör veritabanından bağlam getiriliyor: '{query}'")
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.invoke(query)
    
    context = "\n---\n".join([doc.page_content for doc in relevant_docs])
    
    print("\n[4/4] Yerel Dil Modeli Yükleniyor ve Cevap Üretiliyor...")
    
    # Hafif ve hızlı yerel instruction modeli
    generator = pipeline(
        "text-generation", 
        model="Qwen/Qwen2.5-0.5B-Instruct", 
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    messages = [
        {"role": "system", "content": "Sen yardımsever bir asistansın. Sadece verilen metin bağlamına dayanarak sorulara kısa ve doğru cevaplar ver."},
        {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {query}"}
    ]
    
    output = generator(messages, max_new_tokens=150)
    response = output[0]["generated_text"][-1]["content"]
    
    print("\n" + "="*50)
    print("🤖 YEREL MODELİN CEVABI:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    sample_file = "bilgi.txt"
    chunks = load_and_split_docs(sample_file)
    vector_db = create_vector_db(chunks)
    
    sorgu = "Proje teslimi için neler gereklidir?"
    generate_answer(vector_db, sorgu)