# 🏢 Enterprise HR Query Resolution System

An AI-powered internal HR assistant built with **LangGraph** + **Claude (Anthropic API)** that answers employee queries from organizational documents, escalates complex requests, and provides HR analytics.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HR QUERY RESOLUTION SYSTEM                       │
│                                                                       │
│  Employee Query                                                       │
│       │                                                               │
│       ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                  LangGraph Pipeline                          │     │
│  │                                                             │     │
│  │  [1. Classify Query] ──► [2. Retrieve Docs (RAG)] ──►      │     │
│  │         │                      ChromaDB                     │     │
│  │         │                                                   │     │
│  │  (Sensitive?)                                               │     │
│  │     YES ──► [Escalate Immediately]                         │     │
│  │         │                                                   │     │
│  │         ▼                                                   │     │
│  │  [3. Generate Response]                                     │     │
│  │         │           (Claude LLM + Context)                 │     │
│  │         ▼                                                   │     │
│  │  [4. Assess Confidence]                                     │     │
│  │         │                                                   │     │
│  │  (Low Confidence?) ──YES──► [5. Handle Escalation]        │     │
│  │         │                                                   │     │
│  │         ▼                                                   │     │
│  │  [6. Log Analytics] ──► SQLite DB                          │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌─────────────────┐    ┌──────────────────────────────────────┐     │
│  │  FastAPI Server │    │   Plotly Dash Analytics Dashboard    │     │
│  │  - /query       │    │   - Query trends                     │     │
│  │  - /documents   │    │   - FAQ patterns                     │     │
│  │  - /analytics   │    │   - Department distribution          │     │
│  │  - /escalations │    │   - Escalation tracking              │     │
│  └─────────────────┘    └──────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **LangGraph Multi-Step Pipeline** | 6-node graph: classify → retrieve → generate → assess → escalate → log |
| 📄 **Large Document Understanding** | Ingests PDF, DOCX, TXT with intelligent chunking via ChromaDB |
| 🎯 **Confidence-Based Escalation** | Auto-escalates when confidence < threshold or query is sensitive |
| 🔐 **Role-Based Access Control** | Different responses for employee/manager/hr_admin/executive |
| 📊 **HR Analytics Dashboard** | Plotly Dash dashboard with real-time trends and FAQ tracking |
| 🗂️ **FAQ Pattern Detection** | Tracks frequently asked questions for HR insights |
| ⚡ **REST API** | FastAPI backend with full Swagger docs |
| 🐳 **Docker Support** | One-command deployment with docker-compose |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd hr_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
nano .env
```

### 3. Add HR Documents

Place your PDF, DOCX, or TXT policy files in `data/documents/`:
```
data/documents/
├── leave_policy.pdf
├── reimbursement_policy.docx
├── health_insurance.pdf
├── employee_handbook.pdf
└── ...
```

### 4. Ingest Documents

```bash
python main.py ingest
```

### 5. Start the System

```bash
# Option A: CLI Chat Interface (for testing)
python main.py chat

# Option B: Start API Server
python main.py server

# Option C: Start Analytics Dashboard
python main.py dashboard

# Option D: Docker (full stack)
docker-compose up -d
```

---

## 💬 CLI Usage

```bash
# Interactive mode
python main.py chat
> Employee ID: EMP001
> Department: Engineering
> Role: employee
> Your Question: How many sick leaves do I get per year?

# Demo mode (runs sample queries)
python main.py demo
```

---

## 🔌 API Usage

### Process a Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many sick leaves am I entitled to?",
    "employee_id": "EMP001",
    "department": "Engineering",
    "role": "employee"
  }'
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "query": "How many sick leaves am I entitled to?",
  "response": "According to our Leave Policy, you are entitled to 12 sick leaves per calendar year...",
  "category": "leave_policy",
  "confidence": 0.87,
  "escalated": false,
  "sources": ["leave_policy.txt"],
  "response_time_ms": 2341
}
```

### Upload a Document (HR Admin only)
```bash
curl -X POST http://localhost:8000/documents/ingest \
  -H "X-Employee-Role: hr_admin" \
  -F "file=@./new_policy.pdf" \
  -F "document_type=policy"
```

### Get Analytics (HR Admin only)
```bash
curl http://localhost:8000/analytics/overview \
  -H "X-Employee-Role: hr_admin"
```

**API Docs:** Visit `http://localhost:8000/docs` for interactive Swagger UI.

---

## 📊 Analytics Dashboard

Access at `http://localhost:8050` after starting the dashboard.

Dashboard shows:
- **Total queries** and daily trends
- **Category distribution** (leave, reimbursement, insurance, etc.)
- **Department-wise** query volume
- **Top FAQ patterns** with frequency bars
- **Escalation rate** and pending escalations
- **Average confidence** and response time

---

## 🔀 LangGraph Escalation Flow

```
Query → Classify
           │
    ┌──────┴──────┐
    │             │
 SENSITIVE     NORMAL
    │             │
 ESCALATE    Retrieve Docs
             │
             Generate Response
             │
             Assess Confidence
             │
    ┌─────────────────────┐
    │                     │
 Confidence < 0.6    Confidence ≥ 0.6
    │                     │
 ESCALATE            Log & Return
```

**Escalation Types:**
- `sensitive` — Harassment, grievances, legal matters → Immediate escalation
- `policy_gap` — No relevant policy found in documents
- `low_confidence` — Answer quality below threshold
- `complex` — Multi-step processes needing human guidance

---

## 🗂️ Project Structure

```
hr_assistant/
├── main.py                          # CLI entry point
├── requirements.txt
├── .env.example                     # Environment config template
├── docker-compose.yml
├── Dockerfile
├── data/
│   ├── documents/                   # 📂 Add HR docs here
│   ├── chroma_db/                   # Vector store (auto-created)
│   └── hr_analytics.db              # Analytics SQLite DB (auto-created)
├── logs/
└── src/
    ├── graphs/
    │   └── hr_query_graph.py        # 🧠 LangGraph pipeline (6 nodes)
    ├── tools/
    │   └── document_ingestion.py    # 📄 PDF/DOCX ingestion → ChromaDB
    ├── api/
    │   └── server.py                # 🔌 FastAPI REST API
    ├── dashboard/
    │   └── analytics_dashboard.py   # 📊 Plotly Dash dashboard
    └── utils/
        └── database.py              # 🗄️ SQLAlchemy models
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *required* | Your Anthropic API key |
| `LLM_MODEL` | `claude-opus-4-6` | Claude model to use |
| `ESCALATION_THRESHOLD` | `0.6` | Confidence below this triggers escalation |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector store location |
| `API_PORT` | `8000` | FastAPI server port |
| `DASHBOARD_PORT` | `8050` | Analytics dashboard port |

---

## 🛡️ Role-Based Access

| Role | Can Query | Upload Docs | View Analytics | Resolve Escalations |
|------|-----------|-------------|----------------|---------------------|
| `employee` | ✅ | ❌ | ❌ | ❌ |
| `manager` | ✅ | ❌ | ❌ | ❌ |
| `hr_admin` | ✅ | ✅ | ✅ | ✅ |
| `hr_manager` | ✅ | ✅ | ✅ | ✅ |
| `executive` | ✅ | ❌ | ✅ | ❌ |

Send `X-Employee-Role` header in API requests to authenticate.

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM Orchestration** | LangGraph + LangChain |
| **LLM** | Anthropic Claude (claude-opus-4-6) |
| **Vector Store** | ChromaDB + SentenceTransformers |
| **API** | FastAPI + Uvicorn |
| **Analytics** | Plotly Dash |
| **Database** | SQLite + SQLAlchemy (async) |
| **Document Parsing** | PyPDF2, python-docx |
| **CLI** | Rich + Typer |

---

## 📝 License

Internal use — ACME Corporation | Built with ❤️ using LangGraph + Claude
