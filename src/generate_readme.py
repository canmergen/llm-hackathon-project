import json
from pathlib import Path
from datetime import datetime

def load_metrics():
    """Load evaluation metrics if available."""
    metrics_path = Path("logs/evaluation_results.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def main():
    """Generate the comprehensive README.md."""
    
    # Load Real Metrics
    data = load_metrics()
    
    # Defaults
    acc = "Hesaplanıyor..."
    recall = "Hesaplanıyor..."
    precision = "Hesaplanıyor..."
    
    if data and "metrics" in data:
        m = data["metrics"]
        # Use Binary Accuracy (Risk vs No Risk) as requested by user
        # This ignores confusion between OK and NA, treating them both as "Compliant"
        if "binary_accuracy" in m:
            acc = f"%{m['binary_accuracy']*100:.1f}"
        elif "accuracy" in m:
            acc = f"%{m['accuracy']*100:.1f}"
        
        if "binary_metrics" in m:
            recall = f"%{m['binary_metrics']['recall']*100:.1f}"
            precision = f"%{m['binary_metrics']['precision']*100:.1f}"
        elif "per_class" in m and "NOT_OK" in m["per_class"]:
            recall = f"%{m['per_class']['NOT_OK']['recall']*100:.1f}"
            precision = f"%{m['per_class']['NOT_OK']['precision']*100:.1f}"

    readme_content = f"""# Banka Dokümanları Uyum ve Denetim Rapor Portalı

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
*   **Güven:** **{{recall}}** risk yakalama oranı (Recall) ile çalışır.
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
| **Risk Yakalama (Recall)** | **{{recall}}** | Hatalı maddelerin kaçını yakaladık? (En kritik metrik) |
| **Model Keskinliği (Precision)** | **{{precision}}** | Verdiğimiz alarmların ne kadarı doğru? |
| **Genel Doğruluk (Accuracy)** | **{{acc}}** | Sistemin genel başarım oranı. |

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

```

## Kullanım Kılavuzu

Sistemi çalıştırmak için işletim sisteminize uygun yöntemi seçebilirsiniz:

### 1. Arayüz (Dashboard) ile Kullanım (Kolay Kurulum)
Aşağıdaki dosyalara çift tıklayarak dashboard'u başlatabilirsiniz:

*   **Windows:** `Streamlit_Dashboard_Windows.bat`
*   **Mac:** `Streamlit_Dashboard_Mac.command`
*   **Linux:** `Streamlit_Dashboard_Linux.sh`

Alternatif olarak terminalden çalıştırmak isterseniz:
```bash
streamlit run src/streamlit_compliance_viewer.py
```

### 2. Analiz Pipeline'ını Çalıştırma
Yeni eklenen dokümanları analiz etmek ve veritabanını güncellemek için:

```bash
python main.py
```

---

## Geliştirme Önerileri ve Gelecek Vizyonu

Projenin kapsamını genişletmek ve endüstriyel standartlara taşımak için planlanan geliştirmeler şunlardır:

*   **OCR Entegrasyonu:** Taranmış PDF ve görsel formatındaki eski sözleşmelerin Tesseract veya AWS Textract teknolojileriyle dijitalleştirilip analiz sürecine dahil edilmesi.
*   **Agentic RAG Mimarisi:** Analiz sürecinin tek bir model yerine, "Savcı" (İddia Makamı) ve "Avukat" (Savunma Makamı) rollerini üstlenen çoklu ajan (Multi-Agent) yapısına evrilmesi. Bu sayede gri alanlardaki maddeler için daha dengeli kararlar üretilmesi.
*   **RLHF ile Uzman Geri Bildirimi:** Hukuk departmanındaki uzmanların verdiği "Onay/Red" geri bildirimlerinin toplanarak modelin (Fine-tuning) eğitilmesi ve kurum kültürüne adapte edilmesi.
*   **Kurumsal Entegrasyon:** Tespit edilen riskli maddelerin otomatik olarak JIRA, ServiceNow veya kurum içi risk yönetimi yazılımlarına "İhlal Kaydı" olarak aktarılması.
*   **Çoklu Dil Desteği:** Uluslararası sözleşmelerin analizi için İngilizce başta olmak üzere farklı dillerde mevzuat ve doküman tarama yeteneğinin eklenmesi.
*   **Bulut Tabanlı Ölçeklendirme:** Sistemin Docker konteynerleri haline getirilerek Kubernetes (K8s) üzerinde mikroservis mimarisiyle çalıştırılması.

---

## Proje Yapısı

```
LLM_Hackathon/
├── main.py                         # Pipeline Ana Giriş Noktası (Orchestrator)
├── Streamlit_Dashboard_Mac.command     # Dashboard Başlatıcı (Mac Shortcut)
├── Streamlit_Dashboard_Windows.bat # Dashboard Başlatıcı (Windows)
├── Streamlit_Dashboard_Linux.sh    # Dashboard Başlatıcı (Linux)
├── requirements.txt                # Bağımlılıklar
├── README.md                       # Proje Dokümantasyonu
├── pipeline_architecture.excalidraw# Mimari Çizimi
├── src/                            # Kaynak Kodlar
│   ├── llm_compliance_check.py     # Üretken AI (Gemma) & Kural Motoru
│   ├── retrieval_utils.py          # Hibrit Arama (RAG + BM25 + RRF) Motoru
│   ├── streamlit_compliance_viewer.py # Web Arayüzü (Dashboard)
│   ├── report_generator_pdf.py     # PDF Raporlama Servisi
│   ├── chroma_tool.py              # Vektör Veritabanı Yönetimi
│   ├── evaluate_model.py           # Başarım Ölçümü & Metrik Hesaplama
│   └── generate_readme.py          # Dinamik Dokümantasyon Üretici
├── data/                           # Veri Katmanı
│   ├── banka_dokumanlari/          # Analiz Edilecek Dokümanlar (PDF/Excel)
│   ├── teblig/                     # Mevzuat (Tebliğ) Metinleri
│   └── ground_truth.json           # Doğrulama (Verification) Verisi
├── docs/                           # Raporlar ve Görseller
│   ├── pipeline_design.png         # Akış Diyagramı
│   ├── yonetici_ozeti.png          # Özet Rapor Görseli
│   └── uyum_denetim_raporu.xlsx    # Yöneticiler için Excel Çıktısı
└── logs/                           # Sistem Çıktıları
    ├── compliance_results/         # İşlenmiş JSON Sonuçları
    ├── evaluation_results.json     # Güncel Performans Metrikleri
    └── *.log                       # İşlem Kayıtları
```

---

**Geliştirici:** Can Mergen  
**Etkinlik:** LLM Hackathon 2025
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"✅ README.md generated successfully with metrics: Acc={{acc}}, Recall={{recall}}")

if __name__ == "__main__":
    main()
