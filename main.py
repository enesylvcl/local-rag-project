import os
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

# Vektör veritabanının diskte saklanacağı klasör
CHROMA_DIR = "./chroma_db"

# Embedding modeli: çok dilli (Türkçe dahil) metinleri vektöre çeviren model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cevap üreten yerel dil modeli (LLM)
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# 1. Doküman Yükleme ve Parçalama (chunking)
def load_and_split_docs(file_path):
    print(f"\n[1/4] Doküman yükleniyor: {file_path}")
    # Dosya PDF ise PDF okuyucu, değilse düz metin okuyucu kullanılır
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')

    documents = loader.load()
    # Metni en fazla 300 karakterlik parçalara böler; parçalar arasında 40 karakter örtüşme bırakır
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = text_splitter.split_documents(documents)
    print(f"[✓] Doküman {len(chunks)} parçaya bölündü.")
    return chunks


# 2. Vektör Veritabanını Hazırlama (embedding + ChromaDB)
def get_vector_db(file_path):
    # Metinleri vektöre çevirecek embedding modeli
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # chroma_db klasörü zaten varsa, belgeyi tekrar işlemeden diskteki veritabanını yükle
    if os.path.exists(CHROMA_DIR):
        print(f"\n[2/4] Mevcut vektör veritabanı yükleniyor: {CHROMA_DIR}")
        vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        # Yoksa: belgeyi yükle, parçala, vektörlere çevir ve ChromaDB'ye kaydet
        chunks = load_and_split_docs(file_path)
        print("[2/4] Vektör veritabanı oluşturuluyor...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    print("[✓] Vektör veritabanı hazır.")
    return vector_db


# 3. Yerel Dil Modelini Yükleme (program başında yalnızca bir kez çağrılır)
def load_llm():
    print("\n[3/4] Yerel dil modeli yükleniyor...")
    generator = pipeline(
        "text-generation",
        model=LLM_MODEL,
        torch_dtype=torch.float32,  # CPU'da da çalışabilmesi için 32-bit hassasiyet
        device_map="auto"           # GPU varsa GPU'yu, yoksa CPU'yu otomatik seçer
    )
    print("[✓] Model hazır.")
    return generator


# 4. Soruya Cevap Üretme (retrieval + prompt + LLM çağrısı)
def generate_answer(vector_db, generator, query):
    print(f"\n[4/4] Vektör veritabanından bağlam getiriliyor: '{query}'")
    # Soruya anlamca en yakın 2 parçayı (k=2) getirir
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.invoke(query)

    # Bulunan parçaları tek bir bağlam metninde birleştirir
    context = "\n---\n".join([doc.page_content for doc in relevant_docs])

    # Modele gidecek prompt: sistem talimatı + bağlam + soru
    messages = [
        {"role": "system", "content": "Sen yardımsever bir asistansın. Sadece verilen metin bağlamına dayanarak sorulara kısa ve doğru cevaplar ver."},
        {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {query}"}
    ]

    # Model en fazla 150 yeni token üretir
    output = generator(messages, max_new_tokens=150)
    # Çıktıdaki konuşmanın son mesajı modelin cevabıdır
    response = output[0]["generated_text"][-1]["content"]

    print("\n" + "=" * 50)
    print("🤖 YEREL MODELİN CEVABI:")
    print("=" * 50)
    print(response)
    print("=" * 50)


if __name__ == "__main__":
    sample_file = "bilgi.txt"

    # Veritabanı ve model döngüden önce bir kez hazırlanır
    vector_db = get_vector_db(sample_file)
    generator = load_llm()

    # Kullanıcı "q" yazana kadar soru almaya devam eder
    while True:
        sorgu = input("\nSorunuzu yazın (çıkmak için 'q'): ").strip()
        if sorgu.lower() == "q":
            break
        if not sorgu:
            continue  # Boş giriş yapılırsa tekrar sor
        generate_answer(vector_db, generator, sorgu)
