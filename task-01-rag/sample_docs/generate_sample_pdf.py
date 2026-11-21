"""Utility script to generate a rich, multi-page sample PDF for evaluation and demonstration."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors


def create_sample_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    story = []

    # ---------------- PAGE 1: Overview & Architecture ----------------
    story.append(Paragraph("Ghanex Tech Engineering Architecture Handbook", title_style))
    story.append(Paragraph("Document Version: 3.2.0 | Classification: Internal Engineering Standard", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Executive Summary & Core Principles", h2_style))
    story.append(Paragraph(
        "This handbook defines the architectural standards, code quality metrics, and deployment protocols "
        "enforced across all software engineering teams at Ghanex Tech. Our primary objective is to maintain "
        "high velocity while ensuring 99.95% system availability and stringent zero-trust data security.",
        body_style
    ))
    story.append(Paragraph(
        "All engineers are expected to adhere to our three fundamental tenets: "
        "(1) Modularity over monoliths, (2) Deterministic observability through structured telemetry, "
        "and (3) Grounded verification in automated intelligence pipelines.",
        body_style
    ))

    story.append(Paragraph("2. Microservices Architecture & API Conventions", h2_style))
    story.append(Paragraph(
        "All backend microservices must expose RESTful APIs adhering to the OpenAPI 3.1 specification. "
        "Service-to-service internal communication utilizes gRPC over HTTP/2 with bidirectional streaming "
        "where throughput exceeds 5,000 requests per second. Every service must implement distributed tracing "
        "with OpenTelemetry, propagating the 'X-Correlation-ID' header across all asynchronous hops.",
        body_style
    ))
    story.append(Paragraph(
        "API rate limiting is strictly enforced at the API Gateway layer: Standard tier clients receive 60 "
        "requests per minute, whereas Enterprise partners receive up to 1,200 requests per minute with burst allowance.",
        body_style
    ))

    story.append(PageBreak())

    # ---------------- PAGE 2: Security & Governance ----------------
    story.append(Paragraph("3. Data Governance & Cryptographic Standards", h2_style))
    story.append(Paragraph(
        "Security is non-negotiable across the Ghanex Tech ecosystem. All customer data at rest must be encrypted "
        "using AES-256-GCM. Cryptographic keys are rotated automatically every 90 days via HashiCorp Vault. "
        "All data in transit must enforce TLS 1.3 encryption with Perfect Forward Secrecy (PFS).",
        body_style
    ))
    story.append(Paragraph("Key Compliance and Retention Rules:", body_style))
    story.append(Paragraph("• <b>PII Anonymization:</b> Personally Identifiable Information (PII) must be masked at ingestion before entering analytical or vector databases.", bullet_style))
    story.append(Paragraph("• <b>Audit Trail Retention:</b> Immutable security logs must be retained in cold storage for a minimum of 7 years in compliance with SOC-2 Type II standards.", bullet_style))
    story.append(Paragraph("• <b>Session Expiry:</b> User authentication tokens (JWT) carry a strict 15-minute lifespan and must be refreshed using encrypted refresh cookies with HttpOnly and SameSite=Strict flags.", bullet_style))
    story.append(Paragraph("• <b>Access Reviews:</b> Role-Based Access Control (RBAC) permissions are audited quarterly by the Security Review Board.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Incident Severity Matrix & SLA Guarantees", h2_style))
    
    table_data = [
        ["Severity", "Impact Level", "Initial Response SLA", "Resolution Target"],
        ["Sev-1 (Critical)", "Core service down for >20% users", "< 15 minutes", "< 2 hours"],
        ["Sev-2 (Major)", "Degraded performance or partial outage", "< 30 minutes", "< 6 hours"],
        ["Sev-3 (Moderate)", "Non-critical feature broken", "< 2 hours", "< 24 hours"],
        ["Sev-4 (Minor)", "Cosmetic bug or minor UI glitch", "< 8 hours", "< 72 hours"],
    ]
    t = Table(table_data, colWidths=[100, 180, 110, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ]))
    story.append(t)

    story.append(PageBreak())

    # ---------------- PAGE 3: GenAI & RAG Guidelines ----------------
    story.append(Paragraph("5. GenAI & RAG Implementation Directives", h2_style))
    story.append(Paragraph(
        "When deploying Retrieval-Augmented Generation (RAG) pipelines in production, engineering teams must "
        "observe strict grounding and anti-hallucination guardrails to ensure output reliability and safety.",
        body_style
    ))
    story.append(Paragraph("Mandatory RAG Specifications:", body_style))
    story.append(Paragraph("• <b>Chunking Strategy:</b> Recursive character chunking is mandatory. Recommended chunk size is 1,000 characters with an overlap of 200 characters to balance contextual continuity and embedding efficiency.", bullet_style))
    story.append(Paragraph("• <b>Embedding Model:</b> Dense vector representations must use Google text-embedding-004 (768 dimensions) or equivalent high-precision dense embedders. Embeddings must be L2-normalized prior to cosine index insertion.", bullet_style))
    story.append(Paragraph("• <b>Grounding Verification:</b> If the retrieved context chunks contain a cosine similarity below 0.35 or fail to cover the factual predicate of the query, the LLM must explicitly decline to answer with: 'Based on the provided documents, I could not find information regarding [topic].'", bullet_style))
    story.append(Paragraph("• <b>Citation Enforcement:</b> Every factual assertion in generated responses must cite the source document name and page number directly in brackets.", bullet_style))
    story.append(Paragraph("• <b>Prohibited Uses:</b> Automated generation without human-in-the-loop validation is strictly forbidden for production database write migrations and legal contract signing.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("6. Continuous Delivery & Deployment Cadence", h2_style))
    story.append(Paragraph(
        "Production deployments occur exclusively on Tuesday and Thursday mornings between 06:00 and 08:00 UTC "
        "to minimize peak-hour blast radius. Every pull request requires automated test coverage of at least 85% "
        "and approval from two designated code owners before triggering the GitHub Actions deployment pipeline.",
        body_style
    ))

    doc.build(story)
    print(f"Sample PDF successfully generated at: {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).parent / "sample_policy.pdf"
    create_sample_pdf(str(out_file))
