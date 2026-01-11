# Banka Dokümanları Uyum ve Denetim Rapor Portalı

> **LLM Hackathon** kapsamında geliştirilen, BDDK mevzuatı ile banka sözleşmeleri arasındaki uyumu **Yapay Zeka (LLM)** ve **Hibrit Arama (RAG)** teknolojileriyle denetleyen otonom sistemdir.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=flat&logo=streamlit)
![LLM](https://img.shields.io/badge/LLM-Gemma%203%20(27b)-purple?style=flat&logo=ollama)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=flat)

> ⚠️ **Veri Gizliliği Notu:** Bu projede kullanılan tüm veri setleri, Hackathon kapsamında sağlanan kamuya açık (public) dokümanlar ve sentetik (dummy) verilerden oluşmaktadır. Çalışma, herhangi bir gerçek müşteri verisi veya gizli kurumsal veri içermemektedir.

---

## Proje Vizyonu

**Mevcut Süreç:** 
Bankaların **Müşteri Sözleşmeleri** ve **Ücret Tarifeleri**'nin mevzuata uyumu, uzman hukukçular tarafından manuel olarak kontrol edilmektedir. Bu süreç:
*   **Yavaş:** Bir sözleşme analizi saatler sürer.
*   **Pahalı:** Uzman eforu yüksek maliyetlidir. Hata olursa cezası da yüksektir.
*   **Riskli:** İnsan hatasına açıktır.

**Çözüm:**
**Uyum Denetim Portalı**, bu süreci otonom hale getirir.
*   **Hız:** 200 sayfalık sözleşmeyi dakikalar içinde tarar.
*   **Güven:** **%100.0** risk yakalama oranı (Recall) ile çalışır.
*   **ROI (Yatırım Getirisi):** Finansal etki ve yatırım getirisi analizleri sunar.

---

## Temel Özellikler

### 1. Dual Indexing & Hibrit Arama (RAG v2.0)
Sistem, klasik RAG yaklaşımlarının ötesine geçerek iki farklı indeksleme teknolojisini hibrit olarak kullanır:

| Teknoloji | Algoritma | Görevi | Neden Gerekli? |
|---|---|---|---|
| **Semantik Arama** | `ChromaDB` + `bge-m3` | Kavramsal Eşleşme | "Ek ücret" araması ile "Kart aidatı"nı bulabilmek için. |
| **Anahtar Kelime** | `BM25 (Best Matching)` | Tam Eşleşme | "Madde 12/A" veya "BSMV" gibi spesifik terimleri kaçırmamak için. |

**Çalışma Mantığı:**
1.  Kullanıcı veya Sistem bir sorgu (chunk) gönderir.
2.  Sistem, **Vektör DB** ve **Keyword DB** üzerinde paralel arama yapar.
3.  Sonuçlar **Reciprocal Rank Fusion (RRF)** algoritması ile birleştirilir.
4.  En alakalı ve kanıta dayalı sonuçlar LLM modeline bağlam (context) olarak verilir.

### 2. Hibrit Karar Motoru
*   **Kural Motoru:** Kesin kuralları (Örn: "0 TL olmalı") milisaniyeler içinde denetler.
*   **AI Yargıç (Gemma 3:27B):** Karmaşık maddeleri bir hukukçu gibi yorumlar ve mevzuatla karşılaştırır.

### 3. Yönetici Özeti ve Finansal Analiz
Yöneticiler için büyük resmi gören analizler sunar:
*   **Yönetici Özeti:** Tek bir doğal dil paragrafı ile tüm durumu özetler. (Örn: *"Toplam 2.5M TL risk tespit edildi, %96 başarı sağlandı."*)
*   **Finansal Etki:** Potansiyel ceza riski, önlenen zarar ve ROI (Yatırım Getirisi) hesaplar.
*   **Risk Yoğunluğu:** Hataların hangi belgede yoğunlaştığını gösterir.
    
    ![Yönetici Özeti Raporu](docs/yonetici_ozeti.png)
    
    ![Streamlit Dashboard](docs/streamlit_dashboard.png)

---

## End-to-End Pipeline

Proje, ham verinin işlenmesinden son kullanıcı raporuna kadar kesintisiz bir akış (pipeline) sunar.

![End-to-End Pipeline Design](docs/pipeline_design.png)

> **Mimari İnceleme:** Sistemin detaylı mimari çizimini incelemek için [**pipeline_architecture.excalidraw**](./pipeline_architecture.excalidraw) dosyasına bakabilirsiniz.

### Teknik Detaylar
1.  **Mevzuat İndeksleme:** Sadece referans alınacak yasal metinler (Tebliğler) ChromaDB ve BM25'e indekslenir.
2.  **Sorgulama (Querying):** Banka dokümanları parçalanır ve her bir parça için indekslenen mevzuat içinde arama yapılır.
3.  **Reasoning (Akıl Yürütme):** LLM, sağlanan bağlamı kullanarak *Chain-of-Thought (Zincirleme Düşünce)* yöntemiyle maddeleri analiz eder.
4.  **Reporting:** Sonuçlar yapılandırılmış veri (JSON) olarak saklanır ve anlık olarak dashboard'a yansıtılır.
5.  **Feedback Loop (Chatbot):** Analiz sonuçları *ayrı bir Chroma koleksiyonuna* (`compliance_insights`) indekslenir. Chatbot, hem mevzuatı hem de analiz sonuçlarını kullanarak kullanıcı sorularını yanıtlar.

---

## Performans Metrikleri

Gerçek denetim verileri ile yapılan test sonuçları:

| Metrik | Değer | Anlamı |
|---|---|---|
| **Risk Yakalama (Recall)** | **%100.0** | Hatalı maddelerin kaçını yakaladık? (En kritik metrik) |
| **Model Keskinliği (Precision)** | **%65.0** | Verdiğimiz alarmların ne kadarı doğru? |
| **Genel Doğruluk (Accuracy)** | **%95.1** | Sistemin genel başarım oranı. |

---

## Kurulum

```bash
# 1. Projeyi Klonlayın
git clone https://github.com/canmergen/llm-hackathon-project.git

# 2. Kurulumu Yapın
pip install -r requirements.txt

# 3. Modeli İndirin (Ollama)
ollama pull gemma3:27b

# 4. Pipeline'ı Başlatın
python main.py
```

---

## Proje Yapısı

```
LLM_Hackathon/
├── main.py                    # Pipeline Yöneticisi
├── src/
│   ├── llm_compliance_check.py   # AI Analiz Motoru
│   ├── streamlit_compliance_viewer.py # Dashboard (UI)
│   ├── retrieval_utils.py        # RAG Motoru
│   └── report_generator_pdf.py   # Raporlama Modülü
├── data/
│   ├── banka_dokumanlari/        # İncelenen Sözleşmeler
│   └── teblig/                   # Mevzuat Veritabanı
└── logs/                         # Sonuçlar ve Metrikler
```

---

**Geliştirici:** Can Mergen  
**Etkinlik:** LLM Hackathon 2025
