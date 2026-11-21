# Candidate Submission Guidelines

This document outlines the submission format and instructions for candidates completing the technical assessment for the **GenAI & Open-Source Developer Intern** role at Ghanex Tech.

Please read all guidelines carefully before submitting your work.

---

## Submission Checklist

Before submitting, verify that:

- [ ] All three task repositories (or directories within a single mono-repo) are public and accessible.
- [ ] Each task includes a dedicated `README.md` with setup steps, architecture explanations, and tech stack details.
- [ ] All live application URLs are active and publicly reachable.
- [ ] API keys, personal access tokens, and credentials are removed from git history and managed via environment variables.
- [ ] Your open-source contribution (Issue or Pull Request) is publicly visible.
- [ ] All sections of the template below are filled out thoroughly and accurately.

---

## Submission Template

Copy the markdown template below, fill in your details, and submit it through the designated submission form or via email as instructed by the Ghanex Tech recruitment team.

```markdown
# Internship Assessment Submission - [Your Full Name]

## Candidate Information
- **Full Name:** 
- **Email Address:** 
- **GitHub Profile URL:** 
- **LinkedIn Profile (Optional):** 
- **Submission Date:** 

---

## Task Deliverables

### Task 1: AI Document Assistant (RAG)
- **GitHub Repository URL:** https://github.com/khrishChauhan/ghanextech-technical-assessment/tree/main/task-01-rag
- **Live Deployed Application URL:** [Streamlit Deployment URL]
- **Primary Technologies Used:** Python 3.12, Streamlit, Groq API (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`), Google Gemini API (`gemini-2.0-flash`, `models/text-embedding-004`), FAISS (`faiss-cpu`), LangChain Text Splitters, PyPDF, PDFPlumber.
- **Project Summary:** 
  A production-ready, modular AI Document Assistant employing Retrieval-Augmented Generation (RAG). Documents are parsed page-by-page using PyPDF with a PDFPlumber fallback for complex tabular layouts. Extracted passages undergo hierarchical recursive character splitting (`chunk_size=1000`, `chunk_overlap=200`) designed to capture complete semantic paragraphs without diluting vector representations or exceeding attention windows. Chunks are converted into dense float32 vectors, L2-normalized, and indexed into an in-memory FAISS `IndexFlatIP` store, mathematically guaranteeing that inner products represent exact Cosine Similarity.
  
  The retrieval engine extracts the top-4 most relevant passages with full lineage tracking (`source` file name, `page` number, similarity score). The query and context are dispatched to an ultra-fast LLM (Groq / Gemini) guided by strict system grounding prompts. Hallucinations are actively prevented: if facts are missing from the context, the model explicitly outputs: *"The provided document(s) do not contain information regarding this question."* Responses stream in real-time, accompanied by expandable citation cards showing exact source excerpts and relevance scores.
- **Key Challenges & Solutions:** 
  - *Scanned and Encrypted PDF Edge Cases:* Encountered raster/image-only PDFs and password-protected files that cause standard parsers to crash. Built defensive exception hierarchies (`EncryptedPDFError`, `ScannedPDFError`, `EmptyPDFError`, `CorruptPDFError`) with character-density thresholding (<25 chars) and clear user alerts.
  - *Vector Store Stale Leakage:* Re-indexing documents often leads to residual vectors leaking across sessions. Implemented an explicit `clear_index()` lifecycle that purges both the FAISS index and metadata registry upon new uploads.
  - *Dual LLM Provider Flexibility:* Integrated automated provider detection based on key format (`gsk_` for Groq, `AIza` for Gemini), pairing Groq with fast local dense embeddings and Gemini with `text-embedding-004`.

---

### Task 2: AI Agent with Tool Calling
- **GitHub Repository URL:** https://github.com/khrishChauhan/ghanextech-technical-assessment/tree/main/task-02-ai-agent
- **Live Deployed Application URL:** [Streamlit Deployment URL]
- **Primary Technologies Used:** Python 3.12, Streamlit, Groq Python SDK, Open-Meteo REST & Geocoding APIs, Wikipedia REST & OpenSearch APIs, Python `ast` module.
- **Implemented Tools:** 
  1. `calculator`: Safe mathematical expression evaluator using Python Abstract Syntax Trees (AST). Eliminates code injection by strictly forbidding `eval()`, supporting binary operators (+, -, *, /, //, %, **, ^), unary signs, parentheses, and division-by-zero protection. Input: `expression` (string).
  2. `get_weather`: Live real-time global weather conditions fetching temperature (°C), wind speed, and weather condition summaries via Open-Meteo with geocoding resolution. Input: `city` (string).
  3. `wikipedia_search`: Queries Wikipedia's OpenSearch and REST summary APIs to extract verified encyclopedic overviews, summaries, and reference links. Input: `query` (string).
- **Project Summary:** 
  A multi-turn conversational AI Agent with autonomous reasoning and dynamic tool calling powered by Groq Cloud (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`). The `ToolCallingAgent` uses a ReAct-style reasoning loop: it analyzes user intent, decides whether to respond directly (for greetings and general dialogue) or dispatch one or more external tools, and loops until all required actions are satisfied or `max_iterations=5` is reached.
  
  Conversational memory is sustained across turns within `ToolCallingAgent.messages` according to the OpenAI/Groq function calling specification. Under each agent response, an expandable execution trace (`st.expander("🛠️ Tool Executions")`) renders step-by-step telemetry, including input arguments, millisecond execution durations, and structured output payloads.
- **Key Challenges & Solutions:** 
  - *Safe Math Evaluation without Security Vulnerabilities:* Standard `eval()` allows dangerous arbitrary code execution. Engineered an AST walker that strictly inspects `ast.BinOp`, `ast.UnaryOp`, `ast.Constant`, completely rejecting statements, function calls, imports, and attribute lookups.
  - *Sequential Multi-Tool Execution:* Handled compound questions by managing intermediate tool call states, feeding tool outputs back into the conversation, and allowing the LLM to synthesize a cohesive final answer.
  - *Network Resilience & Error Isolation:* Wrapped external API calls (geocoding, weather, Wikipedia) in dedicated exception handlers returning structured error payloads, enabling the agent to recover gracefully and inform the user without crashing.

---

### Task 3: GitHub Repository AI Analyzer & Open-Source Contribution
- **GitHub Repository URL:** 
- **Live Deployed Application URL:** 
- **Primary Technologies Used:** (e.g., Python, Streamlit, GitHub REST API, Gemini API)
- **Project Summary:** 
  *(Explain how your analyzer ingests repositories, calculates metrics, and generates AI recommendations)*
- **Open-Source Contribution:**
  - **Target Open-Source Repository:** [URL of the open-source project]
  - **Contribution URL (Issue / Pull Request):** [Direct link to your issue or PR]
  - **Contribution Summary:** *(Explain the issue identified, the rationale, and the contribution made)*
- **Key Challenges & Solutions:** 
  *(Describe technical obstacles encountered and how you resolved them)*

---

## AI Collaboration Disclosure

We believe in using AI responsibly as an engineering accelerator. Please briefly describe your collaboration with AI tools during this assessment:
- **AI Tools Used:** (e.g., Claude, ChatGPT, GitHub Copilot, Cursor)
- **How You Leveraged AI:** (e.g., boilerplate generation, debugging error tracebacks, researching API schemas)
- **Verification & Ownership:** 
  *(Confirm that you reviewed, tested, and understood all submitted code, and explain any significant manual refactoring you performed)*

---

## Demonstration Video (Optional)
- **Video Link (Loom, YouTube unlisted, Google Drive):** 
*(A 3-5 minute walkthrough showing your deployed projects and code structure is optional but strongly encouraged)*
```

---

## Evaluation Notice

- Incomplete submissions, broken deployment links, or inaccessible private repositories may directly impact evaluation.
- Shortlisted candidates will be invited to a technical interview to discuss their implementation details, architectural choices, and debugging methodology.
