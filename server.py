"""
FastAPI Backend for HR Query Resolution System
Provides REST API endpoints for:
- Query processing
- Document management  
- Analytics data
- User management
"""

import os
import uuid
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
from dotenv import load_dotenv

# from src.graphs.hr_query_graph import process_hr_query
from hr_query_graph import process_hr_query
# from src.tools.document_ingestion import get_ingestion_tool
from document_ingestion import get_ingestion_tool

load_dotenv()

app = FastAPI(
    title="Enterprise HR Query Assistant API",
    description="AI-powered HR policy Q&A system with escalation and analytics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    employee_id: str = "EMP001"
    department: str = "General"
    role: str = "employee"
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many sick leaves am I entitled to per year?",
                "employee_id": "EMP123",
                "department": "Engineering",
                "role": "employee"
            }
        }


class QueryResponse(BaseModel):
    session_id: str
    query: str
    response: str
    category: str
    confidence: float
    escalated: bool
    escalation_type: str
    sources: List[str]
    response_time_ms: int
    timestamp: str


class FeedbackRequest(BaseModel):
    session_id: str
    satisfied: bool
    feedback_text: Optional[str] = None


class EmployeeInfo(BaseModel):
    employee_id: str
    name: str
    department: str
    role: str = "employee"
    email: Optional[str] = None


# ─── Mock Analytics Data ──────────────────────────────────────────────────────

MOCK_ANALYTICS = {
    "total_queries": 1247,
    "queries_today": 43,
    "avg_confidence": 0.78,
    "escalation_rate": 0.12,
    "category_distribution": {
        "leave_policy": 387,
        "reimbursement": 218,
        "insurance": 156,
        "payroll": 134,
        "onboarding": 98,
        "benefits": 87,
        "remote_work": 76,
        "performance": 54,
        "code_of_conduct": 23,
        "it_policy": 14,
    },
    "department_distribution": {
        "Engineering": 312,
        "Sales": 198,
        "Marketing": 176,
        "HR": 145,
        "Finance": 134,
        "Operations": 112,
        "Product": 89,
        "Legal": 81,
    },
    "daily_trends": [
        {"date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
         "queries": 35 + (i * 3 % 15), "escalations": 4 + (i % 5)}
        for i in range(14, -1, -1)
    ],
    "top_faq": [
        {"question": "How do I apply for casual leave?", "count": 89},
        {"question": "What is the medical reimbursement limit?", "count": 76},
        {"question": "How many sick leaves per year?", "count": 68},
        {"question": "How to claim travel expenses?", "count": 61},
        {"question": "What are WFH policy guidelines?", "count": 54},
        {"question": "When is the performance review cycle?", "count": 47},
        {"question": "How to add dependents to insurance?", "count": 43},
        {"question": "What is the probation period?", "count": 38},
    ],
    "pending_escalations": 14,
    "avg_response_time_ms": 2340,
    "resolution_rate": 0.88,
}


# ─── Role-Based Access Check ──────────────────────────────────────────────────

def check_hr_role(x_employee_role: Optional[str] = Header(None)):
    """Simple RBAC check for HR admin endpoints."""
    hr_roles = {"hr_admin", "hr_manager"}
    if not x_employee_role or x_employee_role.lower() not in hr_roles:
        raise HTTPException(
            status_code=403,
            detail="Access denied. HR Admin or Manager role required."
        )
    return x_employee_role


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Enterprise HR Query Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    ingestion_tool = get_ingestion_tool()
    stats = ingestion_tool.get_collection_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "vector_store": stats,
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process an employee HR query through the LangGraph pipeline.
    Handles classification → RAG retrieval → response → confidence → escalation.
    """
    try:
        result = await process_hr_query(
            query=request.query,
            employee_id=request.employee_id,
            department=request.department,
            role=request.role,
            session_id=request.session_id,
        )

        return QueryResponse(
            **result,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for a query response."""
    # In production, update the QueryLog record in DB
    return {
        "status": "success",
        "message": "Feedback recorded. Thank you!",
        "session_id": feedback.session_id,
    }


@app.post("/documents/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    document_type: str = Form("policy"),
    category: Optional[str] = Form(None),
    _role: str = Depends(check_hr_role),
):
    """
    Upload and ingest an HR policy document.
    Requires HR Admin role (send X-Employee-Role: hr_admin header).
    """
    allowed_types = {".pdf", ".docx", ".doc", ".txt", ".md"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_types}"
        )

    # Save file
    save_path = os.path.join(DOCUMENTS_DIR, file.filename)
    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Ingest into vector store
    ingestion_tool = get_ingestion_tool()
    metadata = {"document_type": document_type}
    if category:
        metadata["category"] = category

    result = ingestion_tool.ingest_document(save_path, metadata=metadata)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return {
        "status": "success",
        "message": f"Document '{file.filename}' ingested successfully",
        "details": result,
    }


@app.post("/documents/ingest-directory")
async def ingest_all_documents(_role: str = Depends(check_hr_role)):
    """Ingest all documents from the documents directory. Requires HR Admin role."""
    ingestion_tool = get_ingestion_tool()
    results = ingestion_tool.ingest_directory()
    return {
        "status": "success",
        "total_ingested": len([r for r in results if r.get("status") == "success"]),
        "total_failed": len([r for r in results if r.get("status") == "error"]),
        "details": results,
    }


@app.get("/documents/stats")
async def get_document_stats():
    """Get statistics about the vector store."""
    ingestion_tool = get_ingestion_tool()
    return ingestion_tool.get_collection_stats()


@app.get("/analytics/overview")
async def get_analytics_overview(_role: str = Depends(check_hr_role)):
    """
    Get HR analytics overview. Requires HR Admin/Manager role.
    Returns query trends, FAQ patterns, escalation stats.
    """
    return MOCK_ANALYTICS


@app.get("/analytics/categories")
async def get_category_analytics(_role: str = Depends(check_hr_role)):
    """Get query distribution by category."""
    return {
        "category_distribution": MOCK_ANALYTICS["category_distribution"],
        "top_faq": MOCK_ANALYTICS["top_faq"],
    }


@app.get("/analytics/trends")
async def get_daily_trends(_role: str = Depends(check_hr_role)):
    """Get daily query trends."""
    return {"daily_trends": MOCK_ANALYTICS["daily_trends"]}


@app.get("/escalations/pending")
async def get_pending_escalations(_role: str = Depends(check_hr_role)):
    """Get pending escalations for HR team."""
    # Mock data — in production, query EscalationLog table
    return {
        "pending": [
            {
                "id": f"ESC{i:04d}",
                "employee_id": f"EMP{100+i}",
                "department": ["Engineering", "Sales", "Marketing"][i % 3],
                "escalation_type": ["complex", "sensitive", "policy_gap"][i % 3],
                "priority": ["high", "medium", "low"][i % 3],
                "created_at": (datetime.now() - timedelta(hours=i*3)).isoformat(),
                "status": "pending",
            }
            for i in range(1, 8)
        ],
        "total": 14,
    }


@app.put("/escalations/{escalation_id}/resolve")
async def resolve_escalation(
    escalation_id: str,
    resolution_notes: str,
    _role: str = Depends(check_hr_role),
):
    """Mark an escalation as resolved."""
    return {
        "status": "success",
        "escalation_id": escalation_id,
        "resolved_at": datetime.utcnow().isoformat(),
        "message": "Escalation resolved successfully",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
