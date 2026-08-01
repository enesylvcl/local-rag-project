# Building Your First Local RAG Application with Foundry Local

Bu proje, yerel dokümanlar üzerindeki verileri dışarıya aktarmadan, tamamen cihaz üzerinde çalışan bir **Local RAG (Retrieval-Augmented Generation)** mimarisiyle sorgulamayı sağlar.

## 🚀 Öne Çıkan Özellikler
- **Gizlilik Odaklı (Local Execution):** Tüm veriler ve dil modeli yerel cihazda çalışır, harici API ihtiyacı duymaz.
- **Vektör Veritabanı:** Metin parçaları anlamsal olarak vektörleştirilip `ChromaDB` üzerinde saklanır.
- **Açık Kaynak Yerel Modeller:** Embedding için `all-MiniLM-L6-v2`, metin üretimi için `Qwen2.5-0.5B-Instruct` modelleri kullanılmıştır.

## 🛠️ Mimari Akış
1. **Document Loading & Chunking:** Metin/PDF dokümanları yüklenir ve `RecursiveCharacterTextSplitter` ile küçük parçalara bölünür.
2. **Vector Embedding:** Parçalar HuggingFace embedding modeli ile vektör alanına dönüştürülür.
3. **Retrieval:** Soruya en yakın bağlamlar (context) ChromaDB'den çekilir.
4. **Generation:** Elde edilen bağlam yerel dil modeline beslenerek doğru ve güvenli yanıt üretilir.

## 💻 Kurulum ve Çalıştırma

```bash
# Projeyi klonlayın
git clone [https://github.com/KULLANICI_ADINIZ/local-rag-project.git](https://github.com/KULLANICI_ADINIZ/local-rag-project.git)
cd local-rag-project

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# Uygulamayı çalıştırın
python main.py