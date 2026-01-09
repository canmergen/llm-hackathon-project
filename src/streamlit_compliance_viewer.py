"""Streamlit app to visualize LLM compliance check results."""
import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from io import BytesIO

import sys
import os

# Add project root to sys.path to enable imports from 'src'
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

st.set_page_config(
    page_title="Banka Uyum Analizi",
    page_icon="📊",
    layout="wide"
)


import base64

def get_base64_image(image_path):
    """Convert image to base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

# --- SHARED HELPERS (Moved to Top Scope) ---

@st.cache_resource
def get_embedding_model():
    """Load and cache the SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    # Hardcoded or from config. Using the one from chroma_tool defaults
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def retrieve_context(query: str, n_results: int = 5) -> str:
    """Retrieve relevant analysis results from Chroma (Tebliğ + Insights)."""
    try:
        import chromadb
        from src.config_loader import config
        
        # 1. Get Embedding
        model = get_embedding_model()
        query_vec = model.encode(query, normalize_embeddings=True).tolist()
        
        # 2. Connect to Chroma
        client = chromadb.PersistentClient(path=config["paths"]["chroma_persist_dir"])
        
        context_text = ""
        
        # A. Search Regulations (Tebliğ) - KEY for "Is it legal?"
        # A. Search Regulations (Tebliğ) - KEY for "Is it legal?"
        try:
            coll_teblig = client.get_collection("teblig_chunks")
            res_teblig = coll_teblig.query(query_embeddings=[query_vec], n_results=n_results)
            
            # DEBUG: Print what we found
            # st.write(f"DEBUG TEBLIG FOUND: {len(res_teblig['documents'][0]) if res_teblig['documents'] else 0}")
            
            if res_teblig["documents"] and res_teblig["documents"][0]:
                context_text += "--- İLGİLİ MEVZUAT (TEBLİĞ) ---\n"
                for i, doc in enumerate(res_teblig["documents"][0]):
                    meta = res_teblig["metadatas"][0][i]
                    context_text += f"{i+1}. {doc} (Madde: {meta.get('madde_no')})\n"
                context_text += "\n"
            else:
                # Fallback search if specific match fails?
                pass
        except Exception as e:
            # st.error(f"MEVZUAT ARAŞTIRMA HATASI: {e}")
            print(f"Teblig search error: {e}")

        # B. Search Analysis Results (Insights) - KEY for "Is there a risk?"
        try:
            coll_insights = client.get_collection("compliance_insights")
            res_insights = coll_insights.query(query_embeddings=[query_vec], n_results=n_results)
            
            if res_insights["documents"] and res_insights["documents"][0]:
                context_text += "--- İLGİLİ BELGE ANALİZİ ---\n"
                for i, doc in enumerate(res_insights["documents"][0]):
                    context_text += f"{i+1}. {doc}\n"
        except (ValueError, Exception):
            pass # Collection might not exist yet
        # C. HYBRID SEARCH (Keyword Match in Loaded Session Results)
        # This acts as a fallback if Vector DB is empty or miss-aligned
        if "compliance_results" in st.session_state and st.session_state.compliance_results:
            results = st.session_state.compliance_results
            keywords = query.lower().split()
            # Simple keyword matching: if any keyword is in chunk_text or reason
            matches = []
            for r in results:
                text = r.get("chunk_text", "").lower()
                reason = r.get("reason", "").lower()
                # Check intersection
                if any(k in text for k in keywords) or any(k in reason for k in keywords):
                    matches.append(r)
            
            # Limit matches
            matches = matches[:5]
            if matches:
                context_text += "\n--- ANALİZ EDİLEN DOKÜMANLARDAN BULGULAR (ANAHTAR KELİME) ---\n"
                for i, m in enumerate(matches):
                    context_text += f"{i+1}. Belge: {m.get('document_type')}\n"
                    context_text += f"   İçerik: {m.get('chunk_text', '')[:300]}...\n"
                    context_text += f"   Risk Analizi: {m.get('reason', '')}\n\n"

        if not context_text:
            return "İlgili mevzuat veya doküman bulgusu bulunamadı. Lütfen daha spesifik sorunuz."
            
        return context_text

    except Exception as e:
        return f"Context retrieval failed: {str(e)}"


def generate_ai_response(prompt, context_text=""):
    """Call the LLM API."""
    try:
        from src.config_loader import config
        import requests

        system_prompt = (
            "Sen 'Banka Mevzuatı ve Doküman Analizi' konusunda uzman bir asistansın. "
            "Görevin, sana verilen bilgi kaynaklarını kullanarak soruları profesyonelce yanıtlamaktır.\n"
            "KURALLAR:\n"
            "1. Asla 'CONTEXT', 'Metin' veya 'Verilen bilgi' gibi ifadeler kullanma. Doğrudan 'Mevzuata göre...', 'Analizlere göre...' veya 'BDDK yönetmeliğine göre...' şeklinde cümlelere başla.\n"
            "2. Analiz sonuçlarında bir risk varsa, nedenini net bir dille açıkla.\n"
            "3. Cevabın sanki bir hukuk müşaviri veya kıdemli denetçi konuşuyormuş gibi profesyonel ve akıcı olsun.\n"
            "4. Cevabın Türkçe olsun.\n\n"
            f"BİLGİ KAYNAKLARI:\n{context_text}\n"
        )
        
        payload = {
            "model": config["llm"]["model_name"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        
        response = requests.post(
            f"{config['llm']['base_url']}/api/chat", 
            json=payload,
            timeout=config['llm'].get('timeout', 120)
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"Model hatası: {str(e)}. Lütfen Ollama servisinin çalıştığından emin olun."


def inject_custom_css():
    """Inject custom CSS for an Enterprise-grade Admin Panel."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        /* --- ROOT VARIABLES (Enterprise Theme) --- */
        :root {
            --primary: #FF6200; /* Corporate Orange - Only for CTA */
            --secondary: #525199; /* Corporate Indigo - For Headers/Accents */
            --primary-text: #172B4D; /* Deep Blue/Grey */
            --muted-text: #6B778C;
            --bg-body: #F4F5F7;
            --bg-panel: #FFFFFF;
            --border: #DFE1E6;
            --radius: 4px;
        }

        /* --- GLOBAL & TYPOGRAPHY --- */
        html, body, [class*="css"] {
            font-family: 'Open Sans', 'Segoe UI', sans-serif;
            color: var(--primary-text);
            font-size: 14px;
        }
        
        .stApp {
            background-color: var(--bg-body);
        }

        /* --- SIDEBAR (CLEAN STYLE) --- */
        section[data-testid="stSidebar"] {
            background-color: #FAFAFA !important; /* Very Light Grey */
            border-right: 1px solid #E6E6E6;
        }
        
        /* Force high contrast text in sidebar */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] strong {
             color: #FF6200 !important; /* Corporate Orange Headers */
             font-weight: 700;
        }
        
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label {
            color: #1F2937 !important; /* Dark Grey Text */
        }
        
        section[data-testid="stSidebar"] .stCaption {
            color: #6B7280 !important; /* Muted Grey for footer */
            font-size: 12px;
        }

        /* --- COMPACT METRICS --- */
        div[data-testid="metric-container"] {
            background-color: var(--bg-panel);
            border: 1px solid var(--border);
            border-left: 4px solid var(--secondary); /* Indigo Accent */
            border-radius: var(--radius);
            padding: 12px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 24px;
            color: var(--primary-text);
            font-weight: 700;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
            font-size: 12px;
        }

        /* --- TABS (Styled Radio Buttons) --- */
        /* Force the specific element container holding the radio to full width */
        div.element-container:has(div[data-testid="stRadio"]) {
            width: 100% !important;
        }

        /* Force ST Radio to Full Width */
        div[data-testid="stRadio"], 
        div[data-testid="stRadio"] > div {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Container for the radio group - GLOBAL SELECTOR */
        div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            justify-content: space-between !important;
            gap: 12px;
            background-color: transparent;
            margin-bottom: 24px;
            width: 100% !important; 
        }
        
        /* FORCE ALL DIRECT CHILDREN TO GROW EQUALLY (Handles wrapper divs) */
        div[role="radiogroup"] > * {
            flex: 1 1 0px !important; /* Force EXACT equality ignoring content width */
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
        }

        /* The individual radio options (buttons) */
        div[role="radiogroup"] label {
            background-color: #FFF5EB !important; /* Inactive: Cream */
            border: 1px solid #FFE4CC !important;
            padding: 12px 6px !important; /* Reduced side padding to prevent overflow */
            border-radius: 6px !important;
            color: #FF6200 !important; /* Inactive Text: Orange */
            font-weight: 600 !important;
            font-size: 16px !important;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            width: 100% !important; /* FILL PARENT */
            height: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            white-space: nowrap !important;
            margin: 0 !important;
        }
        
        /* HIDE THE RADIO CIRCLE (The Dots) - RESTORED */
        div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        
        /* --- PRINT STYLES --- */
        /* --- PRINT STYLES (Targeted) --- */
        /* --- PRINT STYLES (HIDDEN DIV STRATEGY) --- */
        @media print {
            /* 1. HIDE EVERYTHING BY DEFAULT using Visibility */
            /* We cannot use display:none on body/stApp because it removes children from the render tree */
            
            /* --- GLOBAL TYPOGRAPHY: CORPORATE --- */
            @font-face {
                font-family: 'Segoe UI';
                src: local('Segoe UI'), local('Roboto'), local('Arial');
                /* Note: Since we don't have the font files, we rely on local installation or fallback */
            }

            html, body, [class*="css"], .stApp {
                font-family: 'Segoe UI', 'Roboto', sans-serif !important;
            }
            
            h1, h2, h3, h4, h5, h6, .stMarkdown, button, input, textarea {
                font-family: 'Segoe UI', 'Roboto', sans-serif !important;
            }

            body {
                visibility: hidden !important;
            }
            
            /* 2. SPECIFICALLY HIDE STREAMLIT WRAPPERS */
            .stApp {
                visibility: hidden !important;
                /* Optional: reduce height to minimize blank pages if possible, 
                   but absolute positioning of our report usually handles it */
            }
            
            header, footer, .stSidebar, .stDecoration, .stToolbar {
                display: none !important; /* These can be removed safely */
            }

            /* 3. SHOW ONLY OUR PRINT CONTAINER */
            #printable-report-container, #printable-report-container * {
                visibility: visible !important;
            }
            
            /* 4. POSITION REPORT TO FILL PAGE */
            #printable-report-container {
                position: absolute !important;
                left: 0 !important;
                top: 0 !important;
                width: 100% !important;
                min-height: 100vh !important;
                margin: 0 !important;
                padding: 20px !important;
                background-color: white !important;
                z-index: 999999 !important;
            }
            
            @page {
                size: A4;
                margin: 1cm;
            }
        }

        /* --- CHAT INTERFACE CUSTOMIZATION --- */
        
        /* 1. User Message (The Question) -> Cream Background */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background-color: #FFF8E1 !important; /* Cream */
            border: 1px solid #FFE4CC;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        /* 2. Assistant Message (The Answer) -> Orange Text */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background-color: #FFFFFF !important; /* White background for contrast */
            border: 1px solid #EEE;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p,
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) li,
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) h1,
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) h2,
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) h3,
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) span {
            color: #FF6200 !important; /* Corporate Orange Text */
            font-weight: 500;
        }

        /* Active State */
        div[role="radiogroup"] label:has(input:checked) {
            background-color: #FF6200 !important; /* Active BG: Solid Orange */
            color: #FFFFFF !important; /* Active Text: White */
            border-color: #FF6200 !important;
            box-shadow: 0 4px 8px rgba(255, 98, 0, 0.25);
        }
        
        /* Active State - Force ALL inner elements to white */
        div[role="radiogroup"] label:has(input:checked) * {
            color: #FFFFFF !important;
        }
        div[role="radiogroup"] label:has(input:checked) span,
        div[role="radiogroup"] label:has(input:checked) p,
        div[role="radiogroup"] label:has(input:checked) div {
            color: #FFFFFF !important;
        }
        
        /* Fallback for older browsers if needed, or if Streamlit adds a class */
        
        /* --- EXPANDER STYLING (Orange Box) --- */
        /* Target the Header (Summary) */
        div[data-testid="stExpander"] > details > summary {
             background-color: #FF6200 !important; /* Corporate Orange Header */
             color: white !important;
             border-radius: 4px;
             font-weight: 600;
        }
        
        div[data-testid="stExpander"] > details > summary:hover {
             background-color: #e55900 !important; /* Slightly darker orange on hover */
             color: white !important;
        }
        
        /* Target the SVG Icon (Chevron) to be White */
        div[data-testid="stExpander"] > details > summary svg {
             fill: white !important;
             color: white !important;
        }

        /* --- UTILS --- */
        hr {
            margin: 1.5em 0;
            border-color: var(--border);
        }
        
        .stButton button {
            border-radius: var(--radius);
        }
        
        /* --- DASHBOARD METRIC CARDS (Matches HTML Report) --- */
        .dashboard-metric-card {
            background: #ffffff;
            border: 1px solid #dfe1e6;
            border-radius: 8px;
            padding: 20px 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .dashboard-metric-value {
             font-size: 40px;
             font-weight: 800;
             margin: 10px 0;
             line-height: 1;
             display: block;
        }
        
        .dashboard-metric-label {
             font-size: 13px;
             color: #5E6C84;
             font-weight: 700;
             text-transform: uppercase;
             letter-spacing: 0.8px;
             margin-bottom: 5px;
        }
        
        .dashboard-metric-sub {
             font-size: 12px;
             color: #42526E;
             font-weight: 500;
             margin-top: 5px;
        }
        
        /* Hide Header */
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def generate_excel_report(data: List[Dict]) -> bytes:
    """Generate a BytesIO Excel file from the results list."""
    if not data:
        return b""
    
    # Flatten/Clean data for Report
    report_rows = []
    for r in data:
        report_rows.append({
            "Doküman Tipi": r.get("document_type"),
            "Madde ID": r.get("chunk_id"),
            "Orijinal Metin": r.get("chunk_text"),
            "Uyum Durumu": r.get("status"),
            "Analiz Gerekçesi": r.get("reason"),
            "İlgili Tebliğ": r.get("citation"),
            "Düzeltme Önerisi": r.get("corrected_text")
        })
    
    df = pd.DataFrame(report_rows)
    
    output = BytesIO()
    # Use xlsxwriter for better formatting if available, else standard
    # Check if xlsxwriter is installed, otherwise fallback to default (openpyxl usually default for pandas)
    try:
        import xlsxwriter
        engine = 'xlsxwriter'
    except ImportError:
        engine = 'openpyxl'

    with pd.ExcelWriter(output, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name='Uyum Raporu')
        # Auto-adjust columns width only if xlsxwriter
        if engine == 'xlsxwriter':
            worksheet = writer.sheets['Uyum Raporu']
            for i, col in enumerate(df.columns):
                worksheet.set_column(i, i, 20)
            
    return output.getvalue()

def display_metric_card(title: str, value: str, subtitle: str, color_accent: str = "#FF6200", val_color: str = "#172B4D"):
    """Display a custom HTML metric card (Dashboard Version)."""
    # Use color_accent for border, val_color for text (default dark)
    # If val_color is explicitly passed as None or same as accent, it can be handled, but default dark is safe for most except TP/FP
    
    html = f"""
    <div class="dashboard-metric-card" style="border-top: 5px solid {color_accent};">
        <div class="dashboard-metric-label">{title}</div>
        <div class="dashboard-metric-value" style="color: {val_color} !important;">{value}</div>
        <div class="dashboard-metric-sub">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_html_box(title: str, content: str, bg_color: str = "#ffffff", border_color: str = "#ccc", text_color: str = "#333"):
    """Display a custom HTML content box instead of st.text_area."""
    # Ensure no code blocks inside content are interfering
    safe_content = content.replace("```", "")
    
    html = f"""
    <div class="content-box" style="background-color: {bg_color}; border-left-color: {border_color}; color: {text_color};">
        <span class="box-title" style="color: {border_color};">{title}</span>
        <div class="box-content">{safe_content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def load_compliance_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load compliance results from JSON file."""
    if not file_path.exists():
        return []
    
    with file_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Add document type based on chunk_id
    for result in results:
        chunk_id = result.get("chunk_id", "").lower()
        
        # Extract document name from chunk_id
        if "kredi_karti" in chunk_id or "kredi" in chunk_id and "karti" in chunk_id:
            doc_type = "Kredi Kartı Sözleşmesi"
        elif "bireysel" in chunk_id or "bankacilik" in chunk_id:
            doc_type = "Bireysel Bankacılık Sözleşmesi"
        elif "ucret" in chunk_id or "tarife" in chunk_id:
            doc_type = "Ücret Tarifesi"
        else:
            doc_type = "Diğer"
        
        result["document_type"] = doc_type
        result["document_name"] = result.get("chunk_id", "").split("_unit_")[0] if "_unit_" in result.get("chunk_id", "") else result.get("chunk_id", "")
    
    return results


def load_evaluation_metrics() -> Optional[Dict[str, Any]]:
    """Load evaluation metrics from JSON file."""
    eval_path = Path("logs/evaluation_results.json")
    if not eval_path.exists():
        return None
    
    with eval_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth() -> Dict[str, str]:
    """Load ground truth labels into a dictionary."""
    gt_path = Path("data/ground_truth.json")
    if not gt_path.exists():
        return {}
    
    try:
        with gt_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["chunk_id"]: item.get("ground_truth_label", "NA") for item in data}
    except Exception:
        return {}

def calculate_dynamic_metrics(results: List[Dict[str, Any]], gt_map: Dict[str, str]) -> Dict[str, float]:
    """Calculate metrics dynamically for the given results."""
    if not results or not gt_map:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    y_true = []
    y_pred = []
    
    for r in results:
        chunk_id = r.get("chunk_id")
        if chunk_id in gt_map:
            y_true.append(gt_map[chunk_id])
            y_pred.append(r["status"])

    if not y_true:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    # Binary Mapping (OK/NA -> 0, NOT_OK -> 1)
    # We treat OK and NA as "Compliant" (0) and NOT_OK as "Non-Compliant" (1)
    def to_binary(label):
        return 1 if label == "NOT_OK" else 0

    bin_true = [to_binary(L) for L in y_true]
    bin_pred = [to_binary(L) for L in y_pred]

    tp = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 0)

    accuracy = (tp + tn) / len(bin_true) if len(bin_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "matched_count": len(y_true)
    }

def display_summary_metrics(results: List[Dict[str, Any]]) -> None:
    """Display summary metrics dynamically calculated from the input results."""
    # Count statuses in the current filtered selection
    total = len(results)
    ok_count = sum(r["status"] == "OK" for r in results)
    not_ok_count = sum(r["status"] == "NOT_OK" for r in results)
    na_count = sum(r["status"] == "NA" for r in results)
    
    # Dynamic Calculation against Ground Truth
    gt_map = load_ground_truth()
    metrics = calculate_dynamic_metrics(results, gt_map)
    
    st.markdown("### Model Performans Metrikleri")
    
    # If no results or no overlap with GT, explicitly show 0
    if total == 0:
        metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "matched_count": 0}

    # Matches the HTML Report Design
    st.markdown("""
        <h3 style="margin-bottom: 25px; color: #172B4D; border-left: 5px solid #525199; padding-left: 10px; font-size: 20px; display: flex; align-items: center;">
            <span style="font-size: 24px; margin-right: 10px;">📊</span> Performans Göstergeleri (Tahmin / Gerçek Karşılaştırmalı)
        </h3>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # TP: Green Border, Green Text
        display_metric_card("Doğrulanmış Riskler", f"{metrics.get('tp', 0)}", "Gerçekten Hatalı (TP)", "#2e7d32", val_color="#2e7d32")

    with col2:
        # FP: Red Border, Red Text
        display_metric_card("Hatalı Alarmlar", f"{metrics.get('fp', 0)}", "Gereksiz Uyarı (FP)", "#d32f2f", val_color="#d32f2f")

    with col3:
        # Recall: Blue Border, Dark Text (Mockup style)
        display_metric_card("Risk Yakalama", f"{metrics['recall']:.1%}", "Recall (Kaçırmama)", "#1565c0", val_color="#172B4D")
        
    with col4:
        # Accuracy: Yellow Border, Dark Text (Mockup style)
        display_metric_card("Genel Doğruluk", f"{metrics['accuracy']:.1%}", "Accuracy (Genel Ayrım)", "#f9a825", val_color="#172B4D")
    
    if metrics.get("matched_count", 0) > 0:
        st.caption(f"*Metrikler, Ground Truth ile eşleşen {metrics['matched_count']} madde üzerinden hesaplanmıştır.*")
    
    st.divider()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Doküman İstatistikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_card("TOPLAM MADDE", str(total), "İncelenen Toplam Parça", "#333")
    with col2:
        display_metric_card("UYUMLU", str(ok_count), "Tebliğe Uygun", "#2e7d32")
    with col3:
        display_metric_card("İHLAL", str(not_ok_count), "Uyumsuz / Riskli", "#d32f2f")
    with col4:
        display_metric_card("KAPSAM DIŞI", str(na_count), "Madde Dışı / Genel", "#757575")


def get_status_label(status: str) -> str:
    """Get formal Turkish label for status."""
    return {
        "OK": "UYUMLU",
        "NOT_OK": "İHLAL VAR",
        "NA": "BELİRSİZ / KAPSAM DIŞI"
    }.get(status, "BİLİNMİYOR")

def get_status_color(status: str) -> str:
    """Get color for status."""
    return {
        "OK": "#2e7d32", 
        "NOT_OK": "#d32f2f", 
        "NA": "#757575"
    }.get(status, "#0288d1")

def display_chunk_card(result: Dict[str, Any], index: int, unique_key: str = None) -> None:
    """Display a single chunk result as a card."""
    status = result["status"]
    label = get_status_label(status)
    color = get_status_color(status)
    
    # Remove backticks from chunk_id
    chunk_id_clean = result['chunk_id']
    
    with st.container():
        # Clean Header - No Emojis, Better Spacing
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"#### Madde {index + 1}")
            st.caption(f"ID: {chunk_id_clean}")
        with col2:
            st.markdown(f"**{result.get('document_type', 'Bilinmiyor')}**")
        with col3:
            # Badge style status
            st.markdown(f"""
            <div style="background-color:{color}; color:white; padding:4px 12px; border-radius:15px; text-align:center; font-weight:bold; font-size:14px; margin-bottom: 5px;">
                {label}
            </div>
            """, unsafe_allow_html=True)

            # Assistant Button (Moved Here)
            key_val = unique_key if unique_key else f"btn_{index}"
            def go_to_assistant(text_snippet):
                st.session_state.active_tab = "Uyum Asistanı"
                st.session_state.nav_radio = "Uyum Asistanı"
                st.session_state.messages.append({"role": "user", "content": f"Şu maddeyi incele: {text_snippet}..."})

            st.button(
                "🏦 Asistana Sor", 
                key=f"ask_{key_val}", 
                help="Bu maddeyi asistana sor",
                type="primary",
                on_click=go_to_assistant,
                args=(result['chunk_text'][:50],),
                use_container_width=True
            )
        
        # Orijinal Metin (Using Custom Box)
        display_html_box(
            "BANKA DOKÜMANI (ORİJİNAL)",
            result["chunk_text"],
            bg_color="#fafafa", 
            border_color="#999"
        )
        
        # Analysis Section
        col1, col2 = st.columns(2)
        
        with col1:
            # Analysis Box
            display_html_box(
                "YAPAY ZEKA ANALİZİ",
                result["reason"],
                bg_color="#eef2f5", # Very light blue-grey
                border_color="#525199" # Corporate Blue
            )
            
            if result.get("citation"):
                display_html_box(
                    "İLGİLİ TEBLİĞ REFERANSI",
                    result["citation"],
                    bg_color="#f0f7f0", # Very light green
                    border_color="#2e7d32"
                )
        
        with col2:
            if status == "NOT_OK" and result.get("corrected_text"):
                display_html_box(
                    "ÖNERİLEN DÜZELTME",
                    result["corrected_text"],
                    bg_color="#fff8e1", # Very light amber
                    border_color="#ff8f00"
                )
        
        # Top matches (Clean text, no emojis)
        if result.get("top_matches"):
            with st.expander(f"En Alakalı {len(result['top_matches'])} Tebliğ Maddesi", expanded=False):
                for i, match in enumerate(result["top_matches"], 1):
                    madde_no = match.get("metadata", {}).get("madde", match.get("madde", "?"))
                    fikra_no = match.get("metadata", {}).get("fikra", match.get("fikra", "?"))
                    
                    st.markdown(f"**{i}. Madde {madde_no} Fıkra {fikra_no}** "
                                f"(Benzerlik: {match.get('similarity', 0):.2%})")
                    st.caption(match['text'])
                    if i < len(result['top_matches']):
                        st.divider()

        if result.get("regenerated_response"):
            with st.expander("LLM Yanıt Detayı (JSON)", expanded=False):
                st.code(result["regenerated_response"], language="json")
        # Assistant Button (Right Aligned)
        st.divider()

        st.divider()

def get_consolidated_results(results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group results by (Document, Text) to provide a clean, non-redundant view."""
    import re
    from collections import defaultdict
    
    def extract_numbers(text):
        if not text: return None, None
        m = re.search(r"Madde\s*(\d+)", text, re.I)
        f = re.search(r"Fıkra\s*(\d+)|\((\d+)\)", text, re.I)
        m_val = m.group(1) if m else None
        f_val = (f.group(1) or f.group(2)) if f else None
        return m_val, f_val

    # Grouping by (Doc Name, unique bank text)
    groups = defaultdict(list)
    for r in results_list:
        doc_name = r.get("document_type", "Diğer")
        key = (doc_name, r["chunk_text"])
        groups[key].append(r)

    consolidated = []
    for (doc_name, text), group in groups.items():
        # Status priority: NOT_OK > NA > OK
        status_vals = [r["status"] for r in group]
        if "NOT_OK" in status_vals:
            final_status = "NOT_OK"
        elif "NA" in status_vals:
            final_status = "NA"
        else:
            final_status = "OK"

        # Unique Clauses (Filter out N/A)
        clauses = set()
        for r in group:
            found_this = False
            if r.get("top_matches"):
                for match in r["top_matches"]:
                    meta = match.get("metadata", {})
                    m = meta.get("madde_no", meta.get("madde"))
                    f = meta.get("fikra_no", meta.get("fikra"))
                    if m is not None and str(m).upper() != "N/A":
                        f_str = f if (f is not None and str(f).upper() != "N/A") else "?"
                        clauses.add(f"Madde {m}/Fıkra {f_str}")
                        found_this = True
                    
                    if not found_this:
                        pid = meta.get("parent_chunk_id", "") or meta.get("unit_id", "")
                        if pid and "MADDE_" in pid.upper():
                            parts = pid.split("_")
                            m_id = parts[1] if len(parts) >= 2 else "?"
                            f_id = parts[2] if len(parts) >= 3 else "?"
                            if m_id != "?":
                                clauses.add(f"Madde {m_id}/Fıkra {f_id}")
                                found_this = True
            
            if not found_this:
                cit = r.get("citation", "")
                if cit and "N/A" not in cit.upper():
                    m_cit, f_cit = extract_numbers(cit)
                    if m_cit:
                        clauses.add(f"Madde {m_cit}/Fıkra {f_cit if f_cit else '?'}")
                        found_this = True

        clause_list = sorted([c for c in clauses if "N/A" not in c.upper()])
        clause_text = ", ".join(clause_list) if clause_list else "Genel Tebliğ Kuralları"

        # Merge Analysis and Corrections
        reasons = list(dict.fromkeys([r["reason"] for r in group if r.get("reason")]))
        corrections = list(dict.fromkeys([r["corrected_text"] for r in group if r.get("corrected_text") and r["corrected_text"] != "Gerekli Değil"]))
        citations = list(dict.fromkeys([r.get("citation", "Yok") for r in group]))

        consolidated.append({
            "chunk_id": group[0]["chunk_id"], # Keep first ID for ref
            "chunk_text": text,
            "document_type": doc_name,
            "status": final_status,
            "reason": " | ".join(reasons),
            "corrected_text": " | ".join(corrections) if corrections else "Gerekli Değil",
            "citation": clause_text,
            "citations_raw": " | ".join(citations)
        })
    
    return consolidated

def generate_html_report(results: List[Dict[str, Any]]) -> str:
    """Generate a self-contained HTML report string for printing."""
    from datetime import datetime
    
    # Metrics
    total = len(results)
    risks = [r for r in results if r['status'] == 'NOT_OK']
    risk_count = len(risks)
    
    # Sort with custom priority: NOT_OK (0) > OK (1) > NA (2)
    priority = {"NOT_OK": 0, "OK": 1, "NA": 2}
    sorted_results = sorted(results, key=lambda x: priority.get(x['status'], 3))
    
    # Calculate Metrics (TP, FP, Recall, Accuracy)
    gt_map = load_ground_truth()
    
    y_true = []
    y_pred = []
    
    # Build vectors
    for r in results:
        cid = r.get("chunk_id")
        if cid and cid in gt_map:
            y_true.append(gt_map[cid])
            y_pred.append(r["status"])
            
    # Logic: OK/NA -> 0 (Compliant), NOT_OK -> 1 (Non-Compliant)
    def to_bin(label): return 1 if label == "NOT_OK" else 0
    
    bin_true = [to_bin(x) for x in y_true]
    bin_pred = [to_bin(x) for x in y_pred]
    
    if bin_true:
        tp = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 0)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(bin_true) if len(bin_true) > 0 else 0
    else:
        tp, fp, recall, accuracy = 0, 0, 0, 0
    
    # --- REPORT METRICS CALCULATIONS ---
    from collections import Counter
    
    # Density Metrics
    top_doc_name = "Belirtilmemiş"
    top_doc_pct = 0
    if risks:
        doc_counts = Counter([r['document_type'] for r in risks])
        top_docs = doc_counts.most_common(1)
        if top_docs:
            top_doc_name = top_docs[0][0]
            top_doc_pct = int((top_docs[0][1] / risk_count) * 100)
    
    # Financial Metrics
    avg_penalty = 125000
    system_cost = 50000
    potential_loss = risk_count * avg_penalty
    actual_recall_val = recall # Calculated above
    prevented_loss = int(potential_loss * actual_recall_val)
    
    missed_loss = 0
    if actual_recall_val > 0:
         total_potential = potential_loss / actual_recall_val
         missed_loss = total_potential - potential_loss
         
    roi = prevented_loss / system_cost if system_cost > 0 else 0

    html = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Banka Uyum Yöneticisi Raporu</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #fff; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 3px solid #FF6200; padding-bottom: 10px; }}
            
            /* Metrics Grid Updated */
            .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
            .metric-card {{ background: #ffffff; border: 1px solid #dfe1e6; border-radius: 8px; padding: 25px 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            /* Value matches the border color (passed dynamically) */
            .metric-value {{ font-size: 42px; font-weight: 800; display: block; margin: 15px 0; line-height: 1; }}
            .metric-label {{ font-size: 13px; color: #5E6C84; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; display: block; }}
            .metric-sub {{ font-size: 13px; color: #42526E; font-weight: 500; display: block; margin-top: 8px; }}
            
            .summary {{ margin-bottom: 30px; padding: 15px; background-color: #fff8e1; border-radius: 5px; border-left: 5px solid #FF6200; }}
            .card {{ border: 1px solid #eee; margin-bottom: 15px; padding: 15px; border-radius: 4px; page-break-inside: avoid; }}
            .card-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; align-items: center; }}
            .badge {{ padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; color: white; }}
            .text-box {{ background-color: #f9f9f9; padding: 10px; font-size: 12px; color: #444; margin-bottom: 8px; font-style: italic; }}
        </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h1 style="color: #172B4D; margin: 0;">Banka Uyum Yöneticisi Raporu</h1>
            <p style="color: #666; margin-top: 5px;">Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
        </div>
        
        <!-- 1. EXECUTIVE SUMMARY (New) -->
        <div style="background: linear-gradient(135deg, #FF6200 0%, #FF8C00 100%); border-radius: 12px; padding: 25px; margin-bottom: 30px; color: white; box-shadow: 0 4px 15px rgba(255,98,0,0.2);">
            <h2 style="margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 10px; font-size: 20px;">⚠️ Yönetici Özet Raporu</h2>
            <p style="font-size: 14px; line-height: 1.6; margin: 0;">
                Yapay zeka modeli, <strong>%{actual_recall_val*100:.1f}</strong> risk yakalama oranı ile denetim gerçekleştirmiştir. 
                Düzeltilmesi gereken <strong>{risk_count}</strong> madde aşağıda önem sırasına göre sıralanmıştır. 
                Model, finansal bazda toplam <strong>{potential_loss/1000000:.1f}M TL</strong> potansiyel risk tespit etmiş, 
                toplam <strong>{missed_loss/1000000:.1f}M TL</strong> potansiyel risk kaçırmış ve 
                <strong>{roi:,.0f} kat</strong> ROI sağlamıştır. 
                Risklerin yoğunlaştığı ana doküman <strong>%{top_doc_pct}</strong> pay ile <strong>'{top_doc_name}'</strong>dir.
            </p>
        </div>

        <!-- 2. FINANCIAL IMPACT (New) -->
        <h3 style="margin-bottom: 15px; color: #172B4D; border-left: 5px solid #2e7d32; padding-left: 10px; font-size: 18px;">
             💰 Finansal Etki Analizi
        </h3>
        <div style="display: flex; gap: 15px; margin-bottom: 30px;">
            <div style="flex: 1; background: #FFFBEB; border: 1px solid #FCD34D; border-radius: 8px; padding: 15px; text-align: center;">
                <span style="display: block; font-size: 11px; font-weight: bold; color: #92400E; text-transform: uppercase;">Potansiyel Risk</span>
                <span style="display: block; font-size: 24px; font-weight: 800; color: #DC2626; margin: 5px 0;">₺{potential_loss:,.0f}</span>
            </div>
            <div style="flex: 1; background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 8px; padding: 15px; text-align: center;">
                <span style="display: block; font-size: 11px; font-weight: bold; color: #065F46; text-transform: uppercase;">Önlenen Zarar</span>
                <span style="display: block; font-size: 24px; font-weight: 800; color: #059669; margin: 5px 0;">₺{prevented_loss:,.0f}</span>
            </div>
            <div style="flex: 1; background: #EFF6FF; border: 1px solid #93C5FD; border-radius: 8px; padding: 15px; text-align: center;">
                <span style="display: block; font-size: 11px; font-weight: bold; color: #1E40AF; text-transform: uppercase;">Yatırım Getirisi</span>
                <span style="display: block; font-size: 24px; font-weight: 800; color: #2563EB; margin: 5px 0;">{roi:,.0f} kat</span>
            </div>
        </div>
        
        <!-- Performance Metrics -->
        <h3 style="margin-bottom: 25px; color: #172B4D; border-left: 5px solid #525199; padding-left: 10px; font-size: 20px; display: flex; align-items: center;">
            <span style="font-size: 24px; margin-right: 10px;">📊</span> Performans Göstergeleri (Tahmin / Gerçek Karşılaştırmalı)
        </h3>
        <div class="metrics-grid">
            <div class="metric-card" style="border-top: 5px solid #2e7d32;">
                <span class="metric-label">Doğrulanmış Riskler</span>
                <span class="metric-value" style="color: #2e7d32;">{tp}</span>
                <span class="metric-sub">Gerçekten Hatalı (TP)</span>
            </div>
            <div class="metric-card" style="border-top: 5px solid #d32f2f;">
                <span class="metric-label">Hatalı Alarmlar</span>
                <span class="metric-value" style="color: #d32f2f;">{fp}</span>
                <span class="metric-sub">Gereksiz Uyarı (FP)</span>
            </div>
            <div class="metric-card" style="border-top: 5px solid #1565c0;">
                <span class="metric-label">Risk Yakalama</span>
                <span class="metric-value" style="color: #172B4D;">{recall:.1%}</span>
                <span class="metric-sub">Recall (Kaçırmama)</span>
            </div>
            <div class="metric-card" style="border-top: 5px solid #f9a825;">
                <span class="metric-label">Genel Doğruluk</span>
                <span class="metric-value" style="color: #172B4D;">{accuracy:.1%}</span>
                <span class="metric-sub">Accuracy (Genel Ayrım)</span>
            </div>
        </div>
        
        <div class="summary">
            <h3 style="margin-top:0; color: #172B4D;">🛡️ Yönetici Aksiyon Çağrısı</h3>
            <p><strong>Toplam İncelenen Madde:</strong> {total}</p>
            <p><strong>Tespit Edilen Riskler:</strong> <span style="color: #d32f2f; font-weight: bold;">{risk_count}</span></p>
            <p>{'Yüksek riskli maddeler tespit edilmiştir. Öncelikli inceleme gerektirir.' if risk_count > 0 else 'Kritik bir risk tespit edilmemiştir.'}</p>
        </div>
        
        <h3 style="border-bottom: 1px solid #ddd; padding-bottom: 5px; color: #172B4D;">📋 Detaylı Denetim Dökümü</h3>
    """
    
    # Sorting is already applied above
    for i, res in enumerate(sorted_results, 1):
        color = "#d32f2f" if res['status'] == 'NOT_OK' else ("#2e7d32" if res['status'] == 'OK' else "#757575")
        label = "İHLAL VAR" if res['status'] == 'NOT_OK' else ("UYUMLU" if res['status'] == 'OK' else "KAPSAM DIŞI")
        
        html += f"""
        <div class="card">
            <div class="card-header">
                <span style="font-weight: bold; font-size: 14px; color: #333;">#{i} {res.get('document_type', 'Belge')} (ID: {res.get('chunk_id')})</span>
                <span class="badge" style="background-color: {color};">{label}</span>
            </div>
            <div class="text-box">
                "{res.get('chunk_text', '')}"
            </div>
            <div style="font-size: 13px; color: #172B4D;">
                <strong>Analiz:</strong> {res.get('reason', 'N/A')}
            </div>
        </div>
        """
        
    html += "</div></body></html>"
    return html

def main():
    # Page config moved to top of file
    inject_custom_css()
    
    # --- SESSION STATE INIT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Merhaba! Ben Mergen Bank Yapay Zeka Uyum Asistanıyım. Size nasıl yardımcı olabilirim?"})

    # 1. LOAD RESULTS FIRST (Moved Up)
    compliance_dir = Path("logs/compliance_results")
    results = [] 
    
    if compliance_dir.exists():
        json_files = sorted([
            f for f in compliance_dir.glob("*.json") 
            if f.name not in ["test_compliance.json", "combined_compliance.json"]
        ])
        
        if json_files:
            raw_results_map = {}
            for json_file in json_files:
                file_results = load_compliance_results(json_file)
                for r in file_results:
                    cid = r.get("chunk_id")
                    if cid:
                        raw_results_map[cid] = r
            
            # Consolidate
            results = get_consolidated_results(list(raw_results_map.values()))
            st.session_state.compliance_results = results

    # --- HEADER: ENTERPRISE TOP BAR ---
    # Load Logo and Icons
    import base64
    logo_b64 = ""
    icon_html_b64 = ""
    icon_excel_b64 = ""
    
    logo_path = "assets/bank-icon-logo-design-vector.jpg"
    icon_html_path = "assets/icon_download_blue.png"
    icon_excel_path = "assets/icon_excel_green.jpg"
    
    if Path(logo_path).exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    if Path(icon_html_path).exists():
        with open(icon_html_path, "rb") as f:
            icon_html_b64 = base64.b64encode(f.read()).decode()
    if Path(icon_excel_path).exists():
        with open(icon_excel_path, "rb") as f:
            icon_excel_b64 = base64.b64encode(f.read()).decode()


    # Generate reports for download
    html_report = generate_html_report(results)
    excel_bytes = generate_excel_report(results)

    # === UNIFIED HEADER WITH BUTTONS SIDE BY SIDE ===
    # Use columns: [Logo+Title] [spacer] [Btn1] [Btn2]
    col_main, col_space, col_btn1, col_btn2 = st.columns([4, 1, 1.2, 1.2])
    
    with col_main:
        st.markdown(f'''
        <div style="display: flex; align-items: center; gap: 20px; padding: 12px 0;">
            <img src="data:image/png;base64,{logo_b64}" style="height: 50px;" alt="Mergen Bank Logo">
            <div style="width: 1px; height: 40px; background: #D1D5DB;"></div>
            <div style="display: flex; flex-direction: column; gap: 2px;">
                <h1 style="font-size: 24px; font-weight: 700; color: #111827; margin: 0;">Banka Dokümanları Uyum ve Denetim Rapor Portalı</h1>
                <span style="font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500;">MERGEN AI TECH • V1.0 ENTERPRISE • LLM Hackathon 2025 by Can Mergen</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Button styling - target by column position (3rd and 4th columns)
    st.markdown(f'''
    <style>
    /* 3rd column = Blue button (Yönetici Özeti) */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) div[data-testid="stDownloadButton"] > button {{
        background: #EFF6FF !important;
        color: #1D4ED8 !important;
        border: 1.5px solid #1D4ED8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 14px 10px 40px !important;
        background-image: url('data:image/png;base64,{icon_html_b64}') !important;
        background-repeat: no-repeat !important;
        background-position: 12px center !important;
        background-size: 20px 20px !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) div[data-testid="stDownloadButton"] > button:hover {{
        background-color: #DBEAFE !important;
        border-color: #1E40AF !important;
    }}
    
    /* 4th column = Green button (Excel Raporu) */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(4) div[data-testid="stDownloadButton"] > button {{
        background: #ECFDF5 !important;
        color: #047857 !important;
        border: 1.5px solid #047857 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 14px 10px 40px !important;
        background-image: url('data:image/jpeg;base64,{icon_excel_b64}') !important;
        background-repeat: no-repeat !important;
        background-position: 12px center !important;
        background-size: 20px 20px !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(4) div[data-testid="stDownloadButton"] > button:hover {{
        background-color: #D1FAE5 !important;
        border-color: #065F46 !important;
    }}
    
    /* Align entire row so buttons match logo height */
    div[data-testid="stHorizontalBlock"]:first-of-type {{
        align-items: center !important;
    }}
    
    /* Push buttons down slightly */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) div[data-testid="stDownloadButton"],
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(4) div[data-testid="stDownloadButton"] {{
        margin-top: 24px !important;
    }}
    </style>
    ''', unsafe_allow_html=True)
    
    with col_btn1:
        st.download_button(
            "Yönetici Özeti",
            data=html_report,
            file_name="yonetici_ozeti.html",
            mime="text/html",
            use_container_width=True
        )
    
    with col_btn2:
        st.download_button(
            "Excel Raporu",
            data=excel_bytes,
            file_name="uyum_denetim_raporu.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.divider()

    # --- RESULTS CHECK ---
    if not results:
        if not compliance_dir.exists():
             st.warning(f"Sonuç dizini bulunamadı: {compliance_dir}")
        else:
             st.warning("Henüz analiz sonucu bulunmuyor. Önce `llm_compliance_check.py` çalıştırın.")
        return
        
    # --- SMART NAVIGATION (Custom State-Controlled Tabs) ---
    # We use st.radio styled as tabs to allow programmatic switching (e.g., from Action Buttons)
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Yönetici Özeti"

    # Use a hidden radio button but styled via CSS to look like tabs
    selected_tab = st.radio(
        "",
        ["Yönetici Özeti", "Detaylı Denetim", "Uyum Asistanı"],
        index=["Yönetici Özeti", "Detaylı Denetim", "Uyum Asistanı"].index(st.session_state.active_tab),
        horizontal=True,
        label_visibility="collapsed",
        key="nav_radio",
        on_change=lambda: st.session_state.update({"active_tab": st.session_state.nav_radio}) # Sync state
    )
    
    
    # --- TAB NAVIGATION LOGIC ---
    
    # 1. EXECUTIVE SUMMARY
    if selected_tab == "Yönetici Özeti":
        # 1. MANAGER ACTION BRIEF (Multi-Target)
        risks = [r for r in results if r['status'] == 'NOT_OK'] 
        
        # Load Real Metrics
        eval_metrics = load_evaluation_metrics()
        
        # Defaults
        val_tp = "N/A" # True Positives (Not OK -> Not OK)
        val_fp = "N/A" # False Positives (Not OK -> OK/NA)
        val_recall = "N/A" # Recall (Not OK)
        val_acc = "N/A" # Overall Accuracy
        val_prec = 0.0 # Precision (Model Confidence)
        
        # Financial Metrics Defaults
        actual_recall_val = 1.0
        
        if eval_metrics and "metrics" in eval_metrics:
            m = eval_metrics["metrics"]
            
            # 4. Overall Accuracy (BINARY: Risk vs Safe)
            if "binary_accuracy" in m:
                val_acc = f"{m['binary_accuracy']*100:.1f}%"
            
            # Binary Metrics (NOT_OK vs OK+NA)
            if "binary_metrics" in m:
                bin_m = m["binary_metrics"]
                
                # RECALL (Risk Catch Rate)
                rec = bin_m.get("recall", 0)
                actual_recall_val = rec
                val_recall = f"{rec*100:.1f}%"
                
                # PRECISION (Model Confidence)
                val_prec = bin_m.get("precision", 0)
                
                # DERIVE COUNTS using Binary Support
                support = 0
                if "per_class" in m and "NOT_OK" in m["per_class"]:
                    support = m["per_class"]["NOT_OK"].get("support", 0)
                
                tp_count = int(round(rec * support))
                val_tp = f"{tp_count}"
                
                if val_prec > 0:
                    total_predicted_not_ok = tp_count / val_prec
                    fp_count = int(round(total_predicted_not_ok - tp_count))
                    val_fp = f"{fp_count}"
                else:
                    val_fp = "0"
        
        if risks:
            from collections import Counter
            doc_counts = Counter([r['document_type'] for r in risks])
            total_doc_counts = Counter([r['document_type'] for r in results])
            
            # Get Top 3 Risky Documents
            top_docs = doc_counts.most_common(3)
            
            # Calculate Financials for Summary
            risk_count = len(risks)
            avg_penalty_per_violation = 125000  # Average BDDK penalty per violation (TL)
            
            potential_loss = risk_count * avg_penalty_per_violation
            prevented_loss = int(potential_loss * actual_recall_val)  # Use actual recall
            
            # Missed Loss Calculation
            missed_loss = 0
            if actual_recall_val > 0:
                 total_potential_real = potential_loss / actual_recall_val
                 missed_loss = total_potential_real - potential_loss
            
            system_cost = 50000  # Annual system cost (TL)
            roi = prevented_loss / system_cost if system_cost > 0 else 0  # Simple multiplier
            
            # 1. Insight: Top Risk Category (Document Type)
            top_doc_name = top_docs[0][0]
            top_doc_count = top_docs[0][1]
            top_doc_pct = int((top_doc_count / risk_count) * 100)
            
            # Professional Executive Summary Card (Single Narratine Paragraph)
            exec_html = f'''<div style="background: linear-gradient(135deg, #FF6200 0%, #FF8C00 100%); border-radius: 12px; padding: 24px 28px; margin-bottom: 24px; box-shadow: 0 8px 30px rgba(255,98,0,0.25);">
                <div style="display: flex; align-items: flex-start; gap: 20px;">
                    <div style="background: rgba(255,255,255,0.2); width: 56px; height: 56px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 28px; flex-shrink: 0;">⚠️</div>
                    <div style="flex-grow: 1;">
                        <h3 style="margin: 0 0 8px 0; color: #FFFFFF; font-size: 20px; font-weight: 700;">Yönetici Özet Raporu</h3>
                        <p style="margin: 0; color: #FFFFFF; font-size: 14px; line-height: 1.6;">
                            Yapay zeka modeli, <strong>%{actual_recall_val*100:.1f}</strong> risk yakalama oranı ile denetim gerçekleştirmiştir. Düzeltilmesi gereken <strong>{risk_count}</strong> madde aşağıda önem sırasına göre sıralanmıştır. Model, finansal bazda toplam <strong>{potential_loss/1000000:.1f}M TL</strong> potansiyel risk tespit etmiş, toplam <strong>{missed_loss/1000000:.1f}M TL</strong> potansiyel risk kaçırmış ve <strong>{roi:,.0f} kat</strong> ROI sağlamıştır. Risklerin yoğunlaştığı ana doküman <strong>%{top_doc_pct}</strong> pay ile <strong>'{top_doc_name}'</strong>dir.
                        </p>
                    </div>
                </div>
            </div>'''
            st.markdown(exec_html, unsafe_allow_html=True)
            
            # Define safe navigation callback
            # Define safe navigation callback
            def go_to_detail(active_doc):
                st.session_state.active_tab = "Detaylı Denetim"
                st.session_state.nav_radio = "Detaylı Denetim"
                # Set filters for Auto-Filter
                st.session_state["filter_status_key"] = ["İhlal / Riskli (NOT_OK)"]
                st.session_state["filter_doc_type_key"] = [active_doc]
                # Clear legacy trigger if exists
                if "filter_doc_type" in st.session_state:
                    del st.session_state.filter_doc_type

            # Professional Action Cards
            cols = st.columns(len(top_docs))
            for idx, (doc_name, count) in enumerate(top_docs):
                total = total_doc_counts.get(doc_name, count)
                ratio = (count / total * 100) if total > 0 else 0
                
                # Determine severity color
                if ratio >= 70:
                    severity_color = "#dc2626"
                    severity_bg = "rgba(220, 38, 38, 0.1)"
                    severity_label = "KRİTİK"
                elif ratio >= 40:
                    severity_color = "#f59e0b"
                    severity_bg = "rgba(245, 158, 11, 0.1)"
                    severity_label = "YÜKSEK"
                else:
                    severity_color = "#FF6200"
                    severity_bg = "rgba(255, 98, 0, 0.1)"
                    severity_label = "ORTA"
                
                with cols[idx]:
                    card_html = f'''<div style="background: white; border: 1px solid #E5E7EB; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                        <div style="background: {severity_bg}; color: {severity_color}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; display: inline-block; margin-bottom: 12px;">{severity_label}</div>
                        <h4 style="margin: 0 0 8px 0; color: #1F2937; font-size: 14px; font-weight: 600;">{doc_name}</h4>
                        <div style="font-size: 32px; font-weight: 800; color: {severity_color}; line-height: 1; margin-bottom: 6px;">{count}<span style="font-size: 16px; color: #9CA3AF;">/{total}</span></div>
                        <p style="margin: 0; color: #6B7280; font-size: 12px;">Risk Oranı: <strong style="color: {severity_color};">{ratio:.0f}%</strong></p>
                    </div>'''
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Action Button
                    st.button(
                        "📋 Detayları İncele", 
                        key=f"btn_act_{idx}", 
                        type="primary", 
                        use_container_width=True,
                        on_click=go_to_detail,
                        args=(doc_name,)
                    )
            
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # 2. KPI CARDS (Evaluation / Performance Mode)
        st.markdown("##### Performans Göstergeleri (Tahmin / Gerçek Karşılaştırmalı)")
        




        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
             display_metric_card("Doğrulanmış Riskler", val_tp, "Gerçekten Hatalı (TP)", "#2e7d32", val_color="#2e7d32")
        
        with k2:
             display_metric_card("Hatalı Alarmlar", val_fp, "Gereksiz Uyarı (FP)", "#d32f2f", val_color="#d32f2f")
        
        with k3:
             display_metric_card("Risk Yakalama", val_recall, "Recall (Kaçırmama)", "#1565c0", val_color="#172B4D")
        
        with k4:
             display_metric_card("Genel Doğruluk", val_acc, "Accuracy (Genel Ayrım)", "#f9a825", val_color="#172B4D")

        # --- FINANCIAL IMPACT ANALYSIS (Simulated) ---
        st.markdown("##### Finansal Etki Analizi")
        

        
        imp1, imp2, imp3 = st.columns(3)
        
        with imp1:
            impact_html1 = f'''<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #f59e0b;">
                <p style="margin: 0 0 8px 0; color: #92400e; font-size: 12px; font-weight: 600; text-transform: uppercase;">Potansiyel Risk Tutarı</p>
                <div style="font-size: 28px; font-weight: 800; color: #dc2626;">₺{potential_loss:,.0f}</div>
                <p style="margin: 8px 0 0 0; color: #78350f; font-size: 11px;">Tespit edilmeseydi oluşabilecek ceza</p>
            </div>'''
            st.markdown(impact_html1, unsafe_allow_html=True)
        
        with imp2:
            impact_html2 = f'''<div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #10b981;">
                <p style="margin: 0 0 8px 0; color: #065f46; font-size: 12px; font-weight: 600; text-transform: uppercase;">Önlenen Zarar</p>
                <div style="font-size: 28px; font-weight: 800; color: #059669;">₺{prevented_loss:,.0f}</div>
                <p style="margin: 8px 0 0 0; color: #064e3b; font-size: 11px;">Erken tespit ile engellenen kayıp</p>
            </div>'''
            st.markdown(impact_html2, unsafe_allow_html=True)
        
        with imp3:
            impact_html3 = f'''<div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #3b82f6;">
                <p style="margin: 0 0 8px 0; color: #1e40af; font-size: 12px; font-weight: 600; text-transform: uppercase;">Yatırım Getirisi (ROI)</p>
                <div style="font-size: 28px; font-weight: 800; color: #1d4ed8;">{roi:,.0f} kat</div>
                <p style="margin: 8px 0 0 0; color: #1e3a8a; font-size: 11px;">Her 1₺ yatırım için {roi:,.0f}₺ tasarruf</p>
            </div>'''
            st.markdown(impact_html3, unsafe_allow_html=True)

        st.markdown("")  # Spacing

        # 3. CRITICAL INTERVENTION LIST (Enhanced)
        col_list, col_chart = st.columns([1.6, 1], gap="medium")
        
        with col_list:
            st.markdown("##### Müdahale Gerektiren Riskler")
            if risks:
                grid_data = pd.DataFrame(risks)
                if 'chunk_id' in grid_data.columns:
                     grid_data['chunk_id'] = grid_data['chunk_id'].apply(lambda x: str(x).replace("unit_", "Md. ").replace("__", ".").title())
                
                # Add Metadata Columns
                # Ensure confidence exists (Fix for KeyError)
                if 'confidence' not in grid_data.columns:
                    grid_data['confidence'] = 0.96 # Default to High
                    
                # Risk Level: High if confidence > 0.95 else Medium
                grid_data['risk_level'] = grid_data['confidence'].apply(lambda x: "High" if float(x) > 0.95 else "Medium")
                grid_data['action_status'] = "Bekliyor"
                
                # Truncate Reason and Clean Tags
                grid_data['short_reason'] = grid_data['reason'].apply(
                    lambda x: x.replace("[HEURISTIC] ", "").replace("[LLM] ", "")[:80] + "..." if len(x) > 80 
                    else x.replace("[HEURISTIC] ", "").replace("[LLM] ", "")
                )
                
                st.dataframe(
                    grid_data[['risk_level', 'document_type', 'chunk_id', 'short_reason', 'action_status']],
                    hide_index=True,
                    use_container_width=True,
                    height=350,
                    column_config={
                        "risk_level": st.column_config.TextColumn("Seviye", width="small"),
                        "document_type": st.column_config.TextColumn("Belge", width="small"),
                        "chunk_id": st.column_config.TextColumn("Md.", width="small"),
                        "short_reason": st.column_config.TextColumn("Risk Gerekçesi (Özet)", width="large"),
                        "action_status": st.column_config.SelectboxColumn("Aksiyon", options=["Bekliyor", "Onaylandı", "Red"], width="small"),
                    }
                )
            else:
                 st.success("Aktif risk bulunmuyor.")

        # 4. CHARTS (High Contrast)
        with col_chart:
            st.markdown("##### Risk Yoğunluk Analizi")
            import altair as alt
            
            # Chart Logic
            df = pd.DataFrame(results)
            chart_df = df.copy()
            
            if not chart_df.empty:
                # Pre-calculate Stacking for Centered Labels using Pandas
                # 1. Aggregation
                agg = chart_df.groupby(['document_type', 'status']).size().reset_index(name='count')
                
                # 2. Sort by Status Order for consistent stacking
                status_order = ['OK', 'NOT_OK', 'NA']
                agg['status'] = pd.Categorical(agg['status'], categories=status_order, ordered=True)
                agg = agg.sort_values(['document_type', 'status'])
                
                # 4. Calculate Cumulative Positions (Start/End) based on COUNT
                agg['x_end'] = agg.groupby('document_type')['count'].cumsum()
                agg['x_start'] = agg['x_end'] - agg['count']
                agg['x_mid'] = (agg['x_start'] + agg['x_end']) / 2
                
                # TOTALS Calculation for Right-Side Label
                totals_df = agg.groupby('document_type', as_index=False)['count'].sum()

                # 5. Build Chart
                base = alt.Chart(agg).encode(
                    y=alt.Y('document_type', title=None, sort='-x'),
                    tooltip=['document_type', 'status', 'count']
                )

                bars = base.mark_bar().encode(
                    x=alt.X('x_start', axis=alt.Axis(title='Madde Sayısı')),
                    x2='x_end',
                    color=alt.Color('status', 
                                    scale=alt.Scale(domain=['OK', 'NOT_OK', 'NA'], range=['#36B37E', '#D32F2F', '#9E9E9E']), # Green, Red, Grey
                                    legend=alt.Legend(title="Durum", orient="bottom"))
                )

                # Segment Count Labels (Inside Bar)
                text = base.mark_text(color='white', fontWeight='bold').encode(
                    x='x_mid',
                    text=alt.Text('count')
                )
                
                # Total Count Labels (Outside Bar)
                total_text = alt.Chart(totals_df).mark_text(
                    align='left',
                    dx=5,
                    color='#172B4D',
                    fontWeight='bold'
                ).encode(
                    y=alt.Y('document_type', title=None, sort='-x'),
                    x='count',
                    text=alt.Text('count')
                )

                final_chart = (bars + text + total_text).properties(height=350)
                st.altair_chart(final_chart, use_container_width=True)

    # 2. DETAILED AUDIT
    elif selected_tab == "Detaylı Denetim":
        st.info("🔍 Denetçi Modülü: Tüm maddeleri buradan inceleyip onaylayabilirsiniz.")

        # --- FILTER TOOLBAR (Only in Detaylı Denetim, expanded by default) ---
        with st.expander("Filtreleme & Görünüm Ayarları", expanded=True):
            f_col1, f_col2, f_col3 = st.columns([1, 1.5, 1.5])
            
            with f_col1:
                search_query = st.text_input("Metin İçi Arama", placeholder="Örn: 'cayma hakkı'...", label_visibility="collapsed")
                
            doc_types = sorted({r.get("document_type", "Unknown") for r in results})
            with f_col2:
                selected_doc_types = st.multiselect("Doküman Tipi", options=doc_types, default=doc_types, placeholder="Doküman Seçiniz...", label_visibility="collapsed", key="filter_doc_type_key")

            status_map = {
                "Uyumlu (OK)": "OK", 
                "İhlal / Riskli (NOT_OK)": "NOT_OK", 
                "Kapsam Dışı (NA)": "NA"
            }
            with f_col3:
                selected_status_labels = st.multiselect("Uyum Durumu", options=list(status_map.keys()), default=list(status_map.keys()), placeholder="Durum Seçiniz...", label_visibility="collapsed", key="filter_status_key")
                
            # Convert labels back to internal codes
            status_filter = [status_map[lbl] for lbl in selected_status_labels]

        # Apply filters
        filtered_results = [
            r for r in results
            if r["status"] in status_filter
            and r.get("document_type", "Unknown") in selected_doc_types
        ]

        if search_query:
            search_lower = search_query.lower()
            filtered_results = [
                r for r in filtered_results
                if search_lower in r["chunk_id"].lower() 
                or search_lower in r["chunk_text"].lower()
                or search_lower in r.get("reason", "").lower()
            ]

        subtab_list, subtab_table = st.tabs(["📋 Kart Görünümü", "📉 Tablo Görünümü"])
        
        with subtab_list:
            # Display Results
            current_display_results = filtered_results

            if not current_display_results:
                st.warning("Kriterlere uygun kayıt bulunamadı.")
            else:
                # GROUP BY DOCUMENT TYPE
                from collections import defaultdict
                grouped = defaultdict(list)
                for r in current_display_results:
                    grouped[r.get("document_type", "Diğer")].append(r)
                
                # Render groups (Expanders)
                for doc_type, items in grouped.items():
                    # If target_doc is set, expand it (it will be the only one anyway)
                    is_expanded = True 
                    
                    with st.expander(f"{doc_type} ({len(items)} Madde)", expanded=is_expanded):
                        for i, result in enumerate(items):
                            display_chunk_card(result, i, unique_key=f"{doc_type}_{i}")

        with subtab_table:
            # Also filter table if needed, or just show all
            render_summary_table(filtered_results)

    # 3. CHAT ASSISTANT
    elif selected_tab == "Uyum Asistanı":
        render_chat_tab()

def render_chat_tab():
    """Render a Professional Chat Interface with Custom Alignment."""
    st.markdown("### Dijital Uyum Asistanı")
    st.caption("Analiz sonuçları ve BDDK mevzuatı hakkında sorularınızı yanıtlar.")

    # Initialize history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Merhaba! Ben Uyum Asistanınız. Riskli maddeleri özetlememi ister misiniz?"})

    # Load Lion Logo for Chat
    lion_html = '<div style="font-size: 24px; margin-right: 10px; font-weight:bold; color: #FF6200;">MERGEN</div>' # Fallback
    chat_logo_path = "assets/bank-icon-logo-design-vector.jpg"
    if Path(chat_logo_path).exists():
        import base64
        with open(chat_logo_path, "rb") as f:
            l_data = base64.b64encode(f.read()).decode()
        lion_html = f'<img src="data:image/png;base64,{l_data}" width="40" height="40" style="margin-right: 10px; border-radius: 50%; background: white; padding: 2px;">'

    # Load User Icon for Chat
    user_html = '<div style="font-size: 24px; margin-left: 10px; font-weight:bold; color:#525199;">Sen</div>' # Fallback
    user_icon_path = "assets/user_icon_chat.png"
    if Path(user_icon_path).exists():
        import base64
        with open(user_icon_path, "rb") as f:
            u_data = base64.b64encode(f.read()).decode()
        user_html = f'<img src="data:image/png;base64,{u_data}" width="40" height="40" style="margin-left: 10px; border-radius: 50%; background: white; padding: 2px;">'

    # Display History (Custom HTML for Alignment)
    # User messages Right, Assistant Left
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px; align-items: flex-end;">
                        <div style="background-color: #FFF8E1; color: #172B4D; padding: 10px 15px; border-radius: 15px 15px 0 15px; max-width: 70%; border: 1px solid #FFE0B2; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                            {msg["content"]}
                        </div>
                        {user_html}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 10px; align-items: flex-end;">
                        {lion_html}
                        <div style="background-color: #FF6200; color: white; padding: 10px 15px; border-radius: 15px 15px 15px 0; max-width: 70%; border: 1px solid #E65100; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                            {msg["content"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Chat Input Logic
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Generate Response (if last message is user)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
         with st.spinner("Mevzuat taranıyor..."):
             try:
                 user_txt = st.session_state.messages[-1]["content"]
                 # Quick fix for demo interaction if 'retrieve_context' fails import
                 context = retrieve_context(user_txt) if 'retrieve_context' in globals() else ""
                 response = generate_ai_response(user_txt, context) if 'generate_ai_response' in globals() else "Fonksiyon hatası."
                 
                 st.session_state.messages.append({"role": "assistant", "content": response})
                 st.rerun()
             except Exception as e:
                 st.error(f"Hata: {e}")
    if len(st.session_state.messages) < 3:
        st.markdown("---")
        st.markdown("**Hızlı Sorular:**")
        cols = st.columns(3)
        
        def quick_ask(txt):
            st.session_state.messages.append({"role": "user", "content": txt})
            st.rerun()

        if cols[0].button("Kredi kartı aidatı yasal mı?", use_container_width=True):
             quick_ask("Kredi kartı aidatı yasal mı?")
        if cols[1].button("En riskli sözleşme hangisi?", use_container_width=True):
             quick_ask("En riskli sözleşme hangisi?")
        if cols[2].button("Cayma hakkı süresi nedir?", use_container_width=True):
             quick_ask("Cayma hakkı süresi nedir?")





def get_status_emoji(status: str) -> str:
    """Get formal indicator for status (No Emojis)."""
    return {
        "OK": "UYUMLU",
        "NOT_OK": "İHLAL",
        "NA": "KAPSAM DIŞI"
    }.get(status, "BELİRSİZ")

def create_dataframe_from_results(consolidated_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Map consolidated results to a professional display DataFrame."""
    df_data = []
    for r in consolidated_results:
        status_text = {
            "OK": "UYUMLU",
            "NOT_OK": "İHLAL VAR",
            "NA": "KAPSAM DIŞI"
        }.get(r["status"], "BELİRSİZ")

        df_data.append({
            "Doküman Adı": r.get("document_type", "Diğer"),
            "Uyum Durumu": status_text,
            "Banka Metni": r["chunk_text"],
            "Eşleşen Mevzuat Maddeleri": r["citation"],
            "Analiz Sebebi": r["reason"],
            "Mevzuat Referansı": r.get("citations_raw", "Yok"),
            "Düzeltme/Aksiyon": r["corrected_text"]
        })
    
    return pd.DataFrame(df_data)
def render_summary_table(filtered_results):
    df = create_dataframe_from_results(filtered_results)
    
    st.subheader("Uyum Analiz Raporu")

    # DataFrame Display (Forced Light Theme via CSS)
    st.dataframe(
        df,
        use_container_width=True,
        height=600,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Toplam Kayıt Sayısı:** {len(filtered_results)}")

    # Download buttons Side-by-Side
    col_d1, col_d2, _ = st.columns([1, 1, 4])
    with col_d1:
        st.download_button(
            "CSV OLARAK İNDİR",
            df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig'),
            "uyumluluk_ozet.csv",
            "text/csv"
        )
    with col_d2:
        from io import BytesIO
        excel_buffer = BytesIO()
        
        # Data is already enriched and cleaned in create_dataframe_from_results
        df_excel = df.copy()
        
        try:
            # Using openpyxl for formatting (should be default with pandas/streamlit)
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_excel.to_excel(writer, index=False, sheet_name='Uyumluluk Raporu')
                workbook = writer.book
                worksheet = writer.sheets['Uyumluluk Raporu']
                
                from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
                
                # 1. Header Styling (Bold + Gray Background)
                header_fill = PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')
                header_font = Font(bold=True, size=12)
                thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), 
                                     top=Side(style='thin'), bottom=Side(style='thin'))
                
                for cell in worksheet[1]: # First row
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal='center', vertical='center')

                # 2. Auto-Width & Text Wrapping
                for col in worksheet.columns:
                    max_length = 0
                    column_letter = col[0].column_letter
                    column_name = col[0].value
                    
                    # Estimate width based on header or content
                    max_length = max(len(str(column_name)), 15)
                    
                    for cell in col:
                        # Apply borders
                        cell.border = thin_border
                        # Apply wrapping for data rows
                        if cell.row > 1:
                            cell.alignment = Alignment(wrap_text=True, vertical='top')
                        
                        # Find max content length (cap for readability)
                        try:
                            val = str(cell.value) if cell.value else ""
                            if len(val) > max_length:
                                max_length = len(val)
                        except:
                            pass
                    
                    # Set column width (Dynamic but capped)
                    # For long columns like 'Banka Metni' or 'Analiz', fix a wider width
                    if column_name in ["Banka Metni", "Analiz Sebebi"]:
                        worksheet.column_dimensions[column_letter].width = 60
                    elif column_name in ["Düzeltme/Aksiyon", "Mevzuat Referansı"]:
                        worksheet.column_dimensions[column_letter].width = 45
                    else:
                        adjusted_width = (min(max_length, 40) + 2)
                        worksheet.column_dimensions[column_letter].width = adjusted_width

        except Exception as e:
            # Fallback for engine to avoid crash
            df_excel.to_excel(excel_buffer, index=False)
            
        st.download_button(
            "EXCEL OLARAK İNDİR",
            excel_buffer.getvalue(),
            "uyumluluk_ozet.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



    # --- FOOTER ---
    st.divider()
    ft_c1, ft_c2 = st.columns([0.8, 0.2])
    with ft_c1:
        st.caption("© 2025 Mergen Bank Tech & Data  |  Bu rapor yapay zeka destekli olarak üretilmiştir. Yasal bağlayıcılığı yoktur.")
    with ft_c2:
        st.caption(
            f"🧠 **Model:** Gemma 2 27b\n"
            "📜 **Prompt:** v3.2 (Strict)\n"
            f"🕒 **{pd.Timestamp.now().strftime('%d-%m %H:%M')}"
        )

if __name__ == "__main__":
    main()

