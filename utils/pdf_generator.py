"""
PDF report generation utilities.
"""
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def generate_pdf_report(details, plot_path=None):
    """
    Generate a PDF report of evaluation results.
    
    Args:
        details (dict): Dictionary of resume evaluations
        plot_path (str, optional): Path to plot image
        
    Returns:
        io.BytesIO: PDF buffer
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(200, height - 50, "ResuMate - Resume Evaluation Report")
    y = height - 80
    pdf.setFont("Helvetica", 12)
    
    # Iterate through resumes
    for name, d in details.items():
        pdf.drawString(50, y, f"📄 Resume: {name}")
        y -= 20
        pdf.drawString(70, y, f"📊 Similarity: {d['similarity']:.2f}%")
        y -= 15
        pdf.drawString(70, y, f"📖 ATS Score: {d['ats_score']:.2f}")
        y -= 15
        pdf.drawString(70, y, f"🔠 Grammar Errors: {d['grammar_errors']}")
        y -= 15
        pdf.drawString(70, y, f"💼 Action Verbs: {d['action_verbs_count']}")
        y -= 15
        pdf.drawString(70, y, f"📜 Word Count: {d['word_count']}")
        y -= 25
        
        # Missing Keywords
        pdf.drawString(50, y, "🔍 Missing Keywords:")
        y -= 15
        if d["missing_keywords"]:
            for kw in d["missing_keywords"]:
                pdf.drawString(70, y, f"- {kw}")
                y -= 12
        else:
            pdf.drawString(70, y, "✅ No missing keywords!")
            y -= 12
        
        y -= 15
        
        # Sections Found
        pdf.drawString(50, y, "📑 Sections Found:")
        y -= 15
        for k, v in d["sections_found"].items():
            pdf.drawString(70, y, f"- {k}: {'✔ Found' if v else '❌ Missing'}")
            y -= 12
        
        y -= 30
        
        # Check if new page needed
        if y < 150:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = height - 50
        
        # Action verbs preview
        if d["action_verbs_list"]:
            pdf.drawString(
                70, y,
                f"📝 Verbs: {', '.join(d['action_verbs_list'][:10])}..."
            )
            y -= 15
    
    # Add plot if available
    if plot_path:
        try:
            pdf.showPage()
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(200, height - 50, "📊 Overview Chart")
            graph = ImageReader(plot_path)
            pdf.drawImage(graph, 50, height - 400, width=500, height=300)
        except Exception:
            pdf.drawString(50, height - 100, "⚠ Failed to embed chart.")
    
    pdf.save()
    buffer.seek(0)
    return buffer
