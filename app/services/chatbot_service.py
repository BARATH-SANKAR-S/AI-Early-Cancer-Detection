"""
Chatbot Service
RAG-based cancer information chatbot with PDF knowledge base + Clinical Oncology Synthesis.
Accurately identifies cancer-related questions, synthesizes trusted medical answers from NCI/WHO
documents and clinical oncology guides, and politely redirects non-cancer inquiries.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from app.services.pdf_indexing_service import PDFIndexingService
from app.services.cancer_knowledge_base import find_best_knowledge_match, ONCOLOGY_KNOWLEDGE
from chatbot_config import (
    PDF_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, TOP_K_RESULTS,
    MIN_RELEVANCE_SCORE, MAX_CONTEXT_LENGTH, MEDICAL_DISCLAIMER,
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL,
    NON_CANCER_RESPONSE, NO_INFO_RESPONSE,
    DIAGNOSIS_REFUSAL, PRESCRIPTION_REFUSAL
)

logger = logging.getLogger(__name__)

# Greeting message for general introductory queries
GREETING_RESPONSE = (
    "👋 **Hello! I am your AI Cancer Information Assistant.**\n\n"
    "I am specialized exclusively in oncology to answer questions regarding:\n"
    "• **Cancer Types & Symptoms**: Lung, Breast, Brain, Skin, Leukemia, Colorectal, Prostate, etc.\n"
    "• **Biology & Staging**: Stage 0 to IV, TNM system, cell mutations, benign vs malignant tumors.\n"
    "• **Treatments & Therapies**: Chemotherapy, Radiation, Immunotherapy, Surgery, Targeted therapies.\n"
    "• **Prevention & Screening**: Risk factors, healthy lifestyle guidelines, mammography, CT scans, biopsies.\n"
    "• **Clinical AI Tools**: Upload Brain MRI/CT, Lung CT, or Skin lesion images for automated analysis.\n\n"
    "How can I assist you with cancer-related information today?"
)


class ChatbotService:
    """
    RAG-based cancer information chatbot service.
    Uses sentence-transformers + ChromaDB for retrieval, paired with clinical oncology synthesis.
    Falls back gracefully when an external LLM is not configured.
    """

    # Comprehensive cancer vocabulary regex patterns
    CANCER_KEYWORDS = [
        # Core oncology concepts
        r'\bcancer\b', r'\bcancers\b', r'\bcancerous\b', r'\btumor\b', r'\btumors\b', r'\btumour\b', r'\btumours\b',
        r'\bmalignan\w*', r'\bbenign\b', r'\bcarcinoma\b', r'\bsarcoma\b', r'\blymphoma\b',
        r'\bleukemia\b', r'\bleukaemia\b', r'\bmelanoma\b', r'\bglioma\b', r'\bmeningioma\b',
        r'\bglioblastoma\b', r'\bastrocytoma\b', r'\badenocarcinoma\b', r'\bsquamous\b',
        r'\bblastom\w*', r'\bmyeloma\b', r'\bneoplasm\b', r'\bneoplastic\b',
        r'\boncolog\w*', r'\boncologist\b', r'\bmetastasis\b', r'\bmetastatic\b', r'\bmetastas\w*',

        # Cancer anatomical locations / types
        r'\blung cancer\b', r'\bbreast cancer\b', r'\bskin cancer\b', r'\bbrain cancer\b',
        r'\bbrain tumor\b', r'\blung tumor\b', r'\bcolon cancer\b', r'\bcolorectal\b',
        r'\bprostate cancer\b', r'\bcervical cancer\b', r'\bovarian cancer\b', r'\bpancreatic cancer\b',
        r'\bthyroid cancer\b', r'\bliver cancer\b', r'\bstomach cancer\b', r'\bgastric cancer\b',
        r'\bbladder cancer\b', r'\bkidney cancer\b', r'\besophageal cancer\b', r'\btesticular cancer\b',
        r'\bbone cancer\b', r'\bblood cancer\b', r'\bhead and neck cancer\b', r'\buterine cancer\b',
        r'\bmelanom\w*', r'\bcarcinoid\b', r'\bpolyp\b', r'\bpolyps\b',

        # Symptoms and warning signs
        r'\bcancer symptom\w*', r'\bwarning sign\w*', r'\bhemoptysis\b', r'\bcoughing up blood\b',
        r'\bunexplained weight loss\b', r'\bbreast lump\b', r'\bskin lesion\b', r'\babnormal mole\b',
        r'\babcde rule\b', r'\bswollen lymph\b', r'\bnight sweats\b', r'\bpetechiae\b',

        # Staging & grading
        r'\bcancer stage\b', r'\bcancer staging\b', r'\bstage [0-4ivx]+\b', r'\btstage\b',
        r'\btnm\b', r'\btumor grade\b', r'\bcancer grade\b', r'\bdifferentiation\b',
        r'\bremission\b', r'\brecurrence\b', r'\bpalliative\b', r'\bsurvival rate\b',

        # Treatments and therapies
        r'\bchemotherap\w*', r'\bchemo\b', r'\bradiation therap\w*', r'\bradiotherap\w*',
        r'\bimmunotherap\w*', r'\btargeted therap\w*', r'\bhormone therap\w*',
        r'\bcheckpoint inhibitor\w*', r'\bcar-t\b', r'\bmastectomy\b', r'\blumpectomy\b',
        r'\bcraniotomy\b', r'\bclinical trial\b', r'\btreatment side effect\w*',
        r'\bantiemetic\b', r'\bneuropathy\b', r'\bneutropenia\b', r'\balopecia\b',

        # Diagnostics, screening, and genetics
        r'\bbiopsy\b', r'\bbiopsies\b', r'\bmammogra\w*', r'\bcolonoscop\w*',
        r'\bpap smear\b', r'\bpsa test\b', r'\bpet scan\b', r'\bldct\b', r'\bct scan\b',
        r'\bmri scan\b', r'\btumor marker\b', r'\bca-125\b', r'\bcea\b', r'\bbrca\b',
        r'\begfr\b', r'\bcarcinogen\w*', r'\boncogene\w*', r'\bhpv\b', r'\bhuman papillomavirus\b',

        # Organizations & system tools
        r'\bnational cancer institute\b', r'\bnci\b', r'\bworld health organization\b',
        r'\bwho cancer\b', r'\bgrad-cam\b', r'\bvit\b', r'\bresnet\b', r'\befficientnet\b',
        r'\bmedical image\b', r'\bearly detection\b', r'\bcancer screen\w*'
    ]

    GREETING_PATTERNS = [
        r'^\s*hi\b', r'^\s*hello\b', r'^\s*hey\b', r'^\s*greetings\b',
        r'^\s*who are you\b', r'^\s*what can you do\b', r'^\s*help\b'
    ]

    def __init__(self):
        # Initialize PDF indexing and ChromaDB
        self.indexing_service = PDFIndexingService(PDF_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL)
        self.collection = self.indexing_service.collection
        self.embedding_model = self.indexing_service.embedding_model

        # Index PDFs if collection is empty
        try:
            if self.collection.count() == 0:
                logger.info("Knowledge base is empty — indexing PDFs now...")
                results = self.indexing_service.index_all_pdfs()
                logger.info(f"Indexed {results.get('total_chunks', 0)} chunks from {results.get('successful', 0)} PDFs")
            else:
                logger.info(f"Knowledge base loaded: {self.collection.count()} chunks")
        except Exception as e:
            logger.error(f"Failed to initialise knowledge base: {e}")

        # Optional LLM
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Try to initialise external LLM (OpenAI or Ollama); gracefully default to local synthesis."""
        try:
            if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=OPENAI_MODEL,
                    api_key=OPENAI_API_KEY,
                    temperature=0.3,
                )
                logger.info("Initialised OpenAI LLM")
            elif LLM_PROVIDER == "local":
                from langchain_community.llms import Ollama
                self.llm = Ollama(model=LOCAL_LLM_MODEL, base_url=LOCAL_LLM_BASE_URL)
                logger.info("Initialised local Ollama LLM")
            else:
                logger.info("Using built-in Clinical Oncology RAG Engine (no external API key required)")
        except Exception as e:
            logger.warning(f"LLM init failed ({e}) — using built-in Clinical Oncology RAG Engine")
            self.llm = None

    # ── Classification ──────────────────────────────────────────────────────

    def is_greeting(self, query: str) -> bool:
        q = query.lower().strip()
        return any(bool(re.search(pat, q)) for pat in self.GREETING_PATTERNS)

    def is_cancer_related(self, query: str) -> bool:
        q = query.lower()
        if self.is_greeting(q):
            return True
        return any(bool(re.search(pat, q)) for pat in self.CANCER_KEYWORDS)

    def is_diagnosis_request(self, query: str) -> bool:
        patterns = [
            r'\bdo i have (cancer|a tumor|melanoma|leukemia)\b',
            r'\bdiagnose (me|this|my condition)\b',
            r'\bis this (cancer|a tumor|malignant)\b',
            r'\binterpret (this|my|the).*(scan|result|image|mri|ct|xray|biopsy)\b',
            r'\btell me if i have cancer\b',
        ]
        q = query.lower()
        return any(bool(re.search(p, q)) for p in patterns)

    def is_prescription_request(self, query: str) -> bool:
        patterns = [
            r'\bprescribe (me|for me)\b',
            r'\bwhat (medicine|medication|drug|pill|dosage|dose) should i take\b',
            r'\bhow many mg of .*(should i take|can i take)\b',
            r'\bgive me a prescription\b',
        ]
        q = query.lower()
        return any(bool(re.search(p, q)) for p in patterns)

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve_relevant_chunks(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant text chunks from ChromaDB with calibrated similarity."""
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            count = self.collection.count()
            if count == 0:
                return []

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    # ChromaDB default metric is squared L2 distance.
                    # For unit vectors: dist = 2*(1 - cos_sim) => cos_sim = 1 - (dist / 2)
                    similarity = max(0.0, 1.0 - (dist / 2.0))

                    if similarity >= MIN_RELEVANCE_SCORE:
                        chunks.append({
                            "text": doc,
                            "pdf_name": meta.get("pdf_name", "Medical Document"),
                            "page": meta.get("page", 0),
                            "relevance_score": similarity,
                        })

            # Sort by highest relevance score
            chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
            return chunks
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    # ── Response Generation ─────────────────────────────────────────────────

    def _build_intelligent_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build an articulate, medically rich, structured answer.
        Combines our curated Clinical Oncology Knowledge Base with intelligent synthesis.
        Provides only direct, high-quality medical explanations without book/PDF metadata citations.
        """
        # Find best matching clinical topic
        matched_topic = find_best_knowledge_match(query)

        response_sections = []

        if matched_topic:
            response_sections.append(matched_topic["content"])
        elif chunks:
            # Synthesize key points from top chunks cleanly without mentioning PDF names
            intro = self._get_topic_intro(query.lower())
            response_sections.append(intro)

            bullet_points = []
            seen_snippets = set()
            for c in chunks[:4]:
                txt = c["text"].strip()
                if "Table of Contents" in txt or "All rights reserved" in txt or len(txt) < 30:
                    continue
                cleaned = self._clean_text(txt)
                # Split into sentences or clean items
                lines = [l.strip() for l in cleaned.split("\n") if len(l.strip()) > 20 and not l.startswith("Source:")]
                for line in lines[:2]:
                    norm = line[:50].lower()
                    if norm not in seen_snippets:
                        seen_snippets.add(norm)
                        clean_line = line.lstrip("•-* ").strip()
                        bullet_points.append(f"• **{clean_line[:60]}...**" if len(clean_line) > 60 else f"• {clean_line}")

            if bullet_points:
                response_sections.append("\n\n#### 📋 Key Clinical Points:\n" + "\n".join(bullet_points[:6]))
        else:
            intro = self._get_topic_intro(query.lower())
            response_sections.append(intro)

        # Practical recommendations footer
        guidance = (
            "\n\n---\n"
            "💡 **Clinical Recommendation**: For suspicious symptoms, personal risk evaluations, "
            "or image analysis, please consult an oncologist or utilize our hospital's **Upload Image & Analyze** tool."
        )
        response_sections.append(guidance)

        return "".join(response_sections)

    def _get_topic_intro(self, q_lower: str) -> str:
        """Return a contextual intro sentence based on the query topic."""
        if any(w in q_lower for w in ["symptom", "sign", "feel", "what does"]):
            return "### 🩺 Cancer Signs & Symptoms Information\nEarly recognition of signs and symptoms is crucial for prompt medical attention and successful clinical intervention."
        if any(w in q_lower for w in ["treat", "therapy", "chemotherapy", "radiation", "surgery"]):
            return "### 💊 Cancer Treatment Overview\nModern oncology treatments are tailored to the cancer type, stage, molecular biomarkers, and individual patient health."
        if any(w in q_lower for w in ["prevent", "risk", "avoid", "reduce"]):
            return "### 🛡️ Cancer Prevention & Risk Reduction\nEvidence-based strategies can help reduce cancer risks through lifestyle modifications, screening, and immunization."
        if any(w in q_lower for w in ["screen", "test", "detect", "early detection"]):
            return "### 🔍 Cancer Screening & Diagnostic Modalities\nScreening programs and diagnostic imaging detect pre-cancerous conditions and early-stage localized lesions."
        return "### ℹ️ Medical Information from Our Oncology Knowledge Base\nHere is clinical information based on trusted National Cancer Institute (NCI) and WHO oncology documents:"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean excessive whitespace, linebreaks, and formatting artifacts."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    def generate_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate a response using external LLM if configured, else the built-in clinical engine."""
        if not self.llm:
            return self._build_intelligent_response(query, chunks)

        try:
            context = "\n\n".join([
                f"From {c['pdf_name']}, page {c['page']}:\n{c['text']}"
                for c in chunks[:5]
            ])
            if len(context) > MAX_CONTEXT_LENGTH:
                context = context[:MAX_CONTEXT_LENGTH] + "..."

            prompt = (
                "You are an empathetic, highly knowledgeable medical oncology assistant for our hospital platform. "
                "Answer the user's cancer question clearly, using the context from trusted NCI and WHO documents "
                "along with standard clinical knowledge. Format your answer with clear markdown headings and bullet points.\n\n"
                f"Context from medical documents:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
            resp = self.llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.warning(f"LLM call failed ({e}) — falling back to internal synthesis")
            return self._build_intelligent_response(query, chunks)

    # ── Citations ───────────────────────────────────────────────────────────

    def format_citations(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        pdf_pages: Dict[str, set] = {}
        for c in chunks[:4]:
            name = c["pdf_name"].replace(".pdf.pdf", ".pdf").replace(".pdf", "").replace("_", " ")
            pdf_pages.setdefault(name, set()).add(c["page"])
        parts = [
            f"[{name}: Page(s) {', '.join(str(p) for p in sorted(pages))}]"
            for name, pages in pdf_pages.items()
        ]
        return " • ".join(parts)

    # ── Main Entry Point ────────────────────────────────────────────────────

    def process_query(self, query: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Process a user query and return a structured response dictionary."""
        try:
            # 1. Greeting check
            if self.is_greeting(query):
                return {
                    "response": GREETING_RESPONSE,
                    "sources": [],
                    "citations": "",
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "is_cancer_related": True,
                    "has_sources": False,
                }

            # 2. Strict Cancer Relevance Classification
            if not self.is_cancer_related(query):
                return {
                    "response": NON_CANCER_RESPONSE,
                    "sources": [],
                    "citations": "",
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "is_cancer_related": False,
                    "has_sources": False,
                }

            # 3. Medical Safety: Diagnosis request
            if self.is_diagnosis_request(query):
                return {
                    "response": DIAGNOSIS_REFUSAL,
                    "sources": [],
                    "citations": "",
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "is_cancer_related": True,
                    "is_diagnosis_request": True,
                    "has_sources": False,
                }

            # 4. Medical Safety: Prescription request
            if self.is_prescription_request(query):
                return {
                    "response": PRESCRIPTION_REFUSAL,
                    "sources": [],
                    "citations": "",
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "is_cancer_related": True,
                    "is_prescription_request": True,
                    "has_sources": False,
                }

            # 5. Retrieve from knowledge base
            chunks = self.retrieve_relevant_chunks(query, top_k=TOP_K_RESULTS)

            # 6. Generate rich answer
            response_text = self.generate_response(query, chunks)

            sources = [
                {
                    "pdf_name": c["pdf_name"],
                    "page": c["page"],
                    "relevance_score": round(c["relevance_score"], 3),
                    "excerpt": c["text"][:150] + "..." if len(c["text"]) > 150 else c["text"]
                }
                for c in chunks[:4]
            ]
            citations = self.format_citations(chunks)

            return {
                "response": response_text,
                "sources": sources,
                "citations": citations,
                "disclaimer": MEDICAL_DISCLAIMER,
                "is_cancer_related": True,
                "has_sources": len(sources) > 0,
            }

        except Exception as e:
            logger.error(f"process_query error: {e}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an issue processing your medical query. Please try rephrasing your cancer-related question.",
                "sources": [],
                "citations": "",
                "disclaimer": MEDICAL_DISCLAIMER,
                "error": str(e),
                "is_cancer_related": True,
                "has_sources": False,
            }

    def get_health_status(self) -> Dict[str, Any]:
        try:
            stats = self.indexing_service.get_collection_stats()
            return {
                "status": "healthy",
                "llm_provider": LLM_PROVIDER if self.llm else "clinical_rag_engine",
                "llm_available": self.llm is not None,
                "knowledge_base": stats,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }