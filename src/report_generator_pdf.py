from fpdf import FPDF
from pathlib import Path
from typing import Dict, Any, List

class CompliancePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Banka Uyum Analizi - Raporu', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(metrics: Dict[str, Any], output_path: Path):
    """Generate a PDF summary of the compliance evaluation."""
    pdf = CompliancePDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Overview
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Genel Doğruluk (Accuracy): {metrics['metrics']['accuracy']:.2%}", ln=True)
    if 'reasoning_score' in metrics and metrics['reasoning_score'] > 0:
        pdf.cell(0, 10, f"Akıl Yürütme Kalitesi (Reasoning): {metrics['reasoning_score']:.2%}", ln=True)
    pdf.ln(5)
    
    # Metrics Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Sınıf", 1)
    pdf.cell(40, 10, "Precision", 1)
    pdf.cell(40, 10, "Recall", 1)
    pdf.cell(40, 10, "F1-Score", 1)
    pdf.ln()
    
    pdf.set_font("Arial", size=12)
    for label in ["OK", "NOT_OK", "NA"]:
        if label in metrics['metrics']:
            m = metrics['metrics'][label]
            pdf.cell(40, 10, label, 1)
            pdf.cell(40, 10, f"{m['precision']:.2%}", 1)
            pdf.cell(40, 10, f"{m['recall']:.2%}", 1)
            pdf.cell(40, 10, f"{m['f1']:.2%}", 1)
            pdf.ln()
            
    pdf.ln(10)
    
    # Errors Section
    if metrics.get('errors'):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Hatalı Tahminler (Örnekler)", ln=True)
        pdf.set_font("Arial", size=10)
        
        for err in metrics['errors'][:10]:  # First 10 errors
            chunk_id = err['chunk_id'][:50] + "..."
            truth = err['truth']
            pred = err['prediction']
            pdf.multi_cell(0, 8, f"ID: {chunk_id}\nGerçek: {truth} | Tahmin: {pred}\n-------------------")
            
    pdf.output(str(output_path))
    print(f"📄 PDF Report saved: {output_path}")
