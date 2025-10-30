from __future__ import annotations
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image as RLImage, PageBreak


def generate_pdf_report(details: dict, plot_path: str | None = None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    h2 = styles['Heading2']
    normal = styles['Normal']
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=10)

    story = []
    story.append(Paragraph('ResuMate - Resume Evaluation Report', title_style))
    story.append(Spacer(1, 12))

    for name, d in details.items():
        story.append(Paragraph(f'📄 Resume: {name}', h2))
        story.append(Spacer(1, 6))

        metrics = [
            ['Similarity', f"{d.get('similarity', 0):.2f}%"],
            ['ATS Score', f"{d.get('ats_score', 0):.2f}"],
            ['Grammar Errors', str(d.get('grammar_errors', 0))],
            ['Action Verbs', str(d.get('action_verbs_count', 0))],
            ['Word Count', str(d.get('word_count', 0))],
        ]
        mtable = Table(metrics, colWidths=[2.2*inch, 3.8*inch])
        mtable.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#1a202c')),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
        ]))
        story.append(mtable)
        story.append(Spacer(1, 8))

        story.append(Paragraph('✅ Matched Skills', styles['Heading3']))
        if d.get('matched_skills'):
            skills_tbl = [[s] for s in d['matched_skills']]
            stbl = Table([['Skill']] + skills_tbl, colWidths=[6*inch])
            stbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(stbl)
        else:
            story.append(Paragraph('No matched skills found.', small))
        story.append(Spacer(1, 6))

        story.append(Paragraph('❌ Missing Skills', styles['Heading3']))
        if d.get('missing_skills'):
            mskills_tbl = [[s] for s in d['missing_skills']]
            mst = Table([['Skill']] + mskills_tbl, colWidths=[6*inch])
            mst.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#c53030')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(mst)
        else:
            story.append(Paragraph('No missing skills.', small))
        story.append(Spacer(1, 6))

        story.append(Paragraph('🔍 Missing Keywords', styles['Heading3']))
        if d.get('missing_keywords'):
            kw_tbl = [[k] for k in d['missing_keywords']]
            kt = Table([['Keyword']] + kw_tbl, colWidths=[6*inch])
            kt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(kt)
        else:
            story.append(Paragraph('No missing keywords.', small))
        story.append(Spacer(1, 6))

        story.append(Paragraph('🔠 Grammar Issues', styles['Heading3']))
        if d.get('grammar_details'):
            grows = [[gd.get('Error', '')[:80], gd.get('Sentence', '')[:140]] for gd in d['grammar_details']]
            gtbl = Table([['Error', 'Context']] + grows, colWidths=[2.5*inch, 3.5*inch])
            gtbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(gtbl)
        else:
            story.append(Paragraph('No grammar issues detected.', small))
        story.append(Spacer(1, 6))

        story.append(Paragraph('📑 Sections Found', styles['Heading3']))
        sec_rows = [[k, 'Found' if v else 'Missing'] for k, v in d.get('sections_found', {}).items()]
        sec_tbl = Table([['Section', 'Status']] + sec_rows, colWidths=[3*inch, 3*inch])
        sec_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(sec_tbl)
        story.append(Spacer(1, 6))

        av = ', '.join(d.get('action_verbs_list', [])[:30])
        if av:
            story.append(Paragraph('📝 Action Verbs Preview', styles['Heading3']))
            story.append(Paragraph(av, small))
            story.append(Spacer(1, 12))

        story.append(PageBreak())

    if plot_path:
        try:
            story.append(Paragraph('📊 Overview Chart', h2))
            story.append(Spacer(1, 8))
            img = RLImage(plot_path, width=6*inch, height=3*inch)
            story.append(img)
        except Exception:
            story.append(Paragraph('⚠ Failed to embed chart.', small))

    doc.build(story)
    buffer.seek(0)
    return buffer
