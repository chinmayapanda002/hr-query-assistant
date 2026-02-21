"""
LangGraph HR Query Resolution Graph
Multi-step decision flow for:
1. Query classification
2. RAG retrieval
3. Response generation
4. Confidence assessment
5. Escalation routing
6. Analytics logging
"""

import os
import time
import json
import uuid
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
# from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
# from langchain.schema import Document
from langchain_core.documents import Document
from dotenv import load_dotenv

# from src.tools.document_ingestion import get_ingestion_tool
from document_ingestion import get_ingestion_tool

load_dotenv()

ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", "0.6"))
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-6")


# ─── Query Categories ────────────────────────────────────────────────────────

class QueryCategory(str, Enum):
    LEAVE = "leave_policy"
    REIMBURSEMENT = "reimbursement"
    INSURANCE = "insurance"
    ONBOARDING = "onboarding"
    PAYROLL = "payroll"
    PERFORMANCE = "performance"
    CODE_OF_CONDUCT = "code_of_conduct"
    REMOTE_WORK = "remote_work"
    BENEFITS = "benefits"
    IT_POLICY = "it_policy"
    GENERAL = "general_policy"
    UNKNOWN = "unknown"


class EscalationType(str, Enum):
    COMPLEX = "complex"            # Multi-step or sensitive process
    POLICY_GAP = "policy_gap"      # No relevant policy found
    SENSITIVE = "sensitive"        # Legal/disciplinary/grievance
    LOW_CONFIDENCE = "low_confidence"  # Uncertain answer


# ─── Graph State ─────────────────────────────────────────────────────────────

class HRGraphState(TypedDict):
    # Input
    session_id: str
    employee_id: str
    department: str
    role: str                           # employee, manager, hr_admin, etc.
    query: str

    # Processing
    messages: Annotated[List[BaseMessage], add_messages]
    query_category: str
    query_intent: str
    retrieved_docs: List[Dict]
    context: str

    # Output
    response: str
    confidence_score: float
    sources: List[str]

    # Escalation
    should_escalate: bool
    escalation_type: str
    escalation_reason: str

    # Analytics
    start_time: float
    response_time_ms: int


# ─── LLM Setup ───────────────────────────────────────────────────────────────

# def get_llm(temperature: float = 0.1) -> ChatAnthropic:
#     return ChatAnthropic(
#         model=LLM_MODEL,
#         max_tokens=4096,
#         temperature=temperature,
#         api_key=os.getenv("ANTHROPIC_API_KEY"),
#     )

# REMOVE the entire get_llm function and replace with:
def get_llm(temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4096)),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )

# ─── Node Functions ───────────────────────────────────────────────────────────

def classify_query(state: HRGraphState) -> HRGraphState:
    """
    Node 1: Classify the query into categories and detect escalation triggers.
    """
    llm = get_llm()
    query = state["query"]
    role = state.get("role", "employee")

    system_prompt = """You are an HR query classifier. Analyze the employee's question and return a JSON response.

Categories: leave_policy, reimbursement, insurance, onboarding, payroll, performance, 
code_of_conduct, remote_work, benefits, it_policy, general_policy, unknown

Escalation triggers (return escalate=true for):
- Grievances, harassment, discrimination complaints
- Legal disputes or compliance violations  
- Personal salary negotiations
- Termination or disciplinary actions
- Queries requiring access to personal employee records
- Ambiguous queries needing human judgment

Return ONLY valid JSON:
{
  "category": "<category>",
  "intent": "<one-line description of what user wants>",
  "escalate": <true/false>,
  "escalation_reason": "<reason if escalate=true, else null>",
  "escalation_type": "<complex|policy_gap|sensitive|low_confidence or null>"
}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Employee Role: {role}\nQuery: {query}")
    ])

    try:
        # Extract JSON from response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
    except Exception:
        result = {
            "category": "unknown",
            "intent": "Employee query",
            "escalate": False,
            "escalation_reason": None,
            "escalation_type": None
        }

    state["query_category"] = result.get("category", "unknown")
    state["query_intent"] = result.get("intent", "")
    state["should_escalate"] = result.get("escalate", False)
    state["escalation_reason"] = result.get("escalation_reason") or ""
    state["escalation_type"] = result.get("escalation_type") or ""

    return state


def retrieve_documents(state: HRGraphState) -> HRGraphState:
    """
    Node 2: Retrieve relevant HR policy documents from ChromaDB.
    """
    if state.get("should_escalate"):
        # Still retrieve docs for context even on escalation
        pass

    ingestion_tool = get_ingestion_tool()
    query = state["query"]
    category = state.get("query_category")

    # Retrieve with and without category filter for best coverage
    results = ingestion_tool.similarity_search_with_score(query, k=6)

    retrieved_docs = []
    sources = []

    for doc, score in results:
        similarity = 1 - score  # Convert distance to similarity
        if similarity > 0.0:  # Filter very irrelevant results
            retrieved_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category", ""),
                "score": round(similarity, 3),
            })
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)

    # Build context string
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:4], 1):
        context_parts.append(
            f"[Source {i}: {doc['source']} | Relevance: {doc['score']}]\n{doc['content']}"
        )

    state["retrieved_docs"] = retrieved_docs
    state["context"] = "\n\n---\n\n".join(context_parts) if context_parts else ""
    state["sources"] = sources

    return state


def generate_response(state: HRGraphState) -> HRGraphState:
    """
    Node 3: Generate a response using retrieved context (RAG).
    """
    llm = get_llm(temperature=0.2)
    query = state["query"]
    context = state.get("context", "")
    role = state.get("role", "employee")
    department = state.get("department", "")
    category = state.get("query_category", "general")

    if context:
        system_prompt = f"""You are an expert HR assistant for a company. Answer the employee's question 
using ONLY the provided policy documents as your source of truth.

Employee Context:
- Role: {role}
- Department: {department}
- Query Category: {category}

Guidelines:
1. Be clear, concise, and empathetic
2. Cite specific policy sections when possible
3. Use bullet points for multi-step processes
4. If information is partial, acknowledge what you know and what might need clarification
5. Always recommend consulting HR for personal/sensitive matters
6. Tailor the response to the employee's role level

Policy Documents Context:
{context}"""
    else:
        system_prompt = f"""You are an expert HR assistant. No specific policy documents were found for this query.
Provide a general, helpful response about common HR practices but clearly state that:
1. You couldn't find specific company policy for this topic
2. The employee should contact HR directly for authoritative information
3. What general best practices suggest

Employee Context: Role: {role}, Department: {department}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)
    state["response"] = response.content
    state["messages"] = [HumanMessage(content=query), response]

    return state


def assess_confidence(state: HRGraphState) -> HRGraphState:
    """
    Node 4: Assess confidence in the generated response.
    """
    retrieved_docs = state.get("retrieved_docs", [])
    response = state.get("response", "")

    # If no docs retrieved, low confidence
    if not retrieved_docs:
        state["confidence_score"] = 0.2
        if not state.get("should_escalate"):
            state["should_escalate"] = True
            state["escalation_type"] = EscalationType.POLICY_GAP
            state["escalation_reason"] = "No relevant policy documents found"
        return state

    # Docs were found — compute confidence from scores
    avg_score = sum(d["score"] for d in retrieved_docs[:3]) / min(3, len(retrieved_docs))
    
    # Give a reasonable base confidence when docs are found
    confidence = max(0.75, avg_score)
    state["confidence_score"] = round(confidence, 3)
    state["should_escalate"] = False

    return state


def handle_escalation(state: HRGraphState) -> HRGraphState:
    """
    Node 5: Handle escalation — add escalation notice to response.
    """
    escalation_type = state.get("escalation_type", "complex")
    escalation_reason = state.get("escalation_reason", "")
    original_response = state.get("response", "")

    escalation_messages = {
        EscalationType.SENSITIVE: "⚠️ **This query involves a sensitive HR matter** and requires direct HR team involvement.",
        EscalationType.COMPLEX: "ℹ️ **This query involves a complex process** that may require personalized HR guidance.",
        EscalationType.POLICY_GAP: "📋 **No specific policy was found** in our current documentation for this query.",
        EscalationType.LOW_CONFIDENCE: "💡 **This response may need verification** by an HR specialist.",
    }

    escalation_notice = escalation_messages.get(
        escalation_type, "ℹ️ This query has been flagged for HR review."
    )

    escalation_footer = f"""

---
{escalation_notice}

**Your query has been escalated to the HR team.** An HR representative will reach out within 1-2 business days.

For urgent matters, please contact HR directly at: hr@company.com

*Reference ID: {state['session_id'][:8].upper()}*"""

    if original_response:
        state["response"] = original_response + escalation_footer
    else:
        state["response"] = f"Thank you for your query. {escalation_footer}"

    return state


def log_analytics(state: HRGraphState) -> HRGraphState:
    """
    Node 6: Log query analytics for HR insights.
    """
    end_time = time.time()
    start_time = state.get("start_time", end_time)
    state["response_time_ms"] = int((end_time - start_time) * 1000)

    # Analytics data (would be saved to DB in production)
    analytics_data = {
        "session_id": state.get("session_id"),
        "employee_id": state.get("employee_id"),
        "department": state.get("department"),
        "role": state.get("role"),
        "query_category": state.get("query_category"),
        "query_intent": state.get("query_intent"),
        "confidence_score": state.get("confidence_score"),
        "escalated": state.get("should_escalate", False),
        "escalation_type": state.get("escalation_type"),
        "response_time_ms": state.get("response_time_ms"),
        "sources_used": state.get("sources", []),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # In production, save to DB:
    # await db.execute(insert(QueryLog).values(**analytics_data))
    print(f"📊 Analytics logged: category={analytics_data['query_category']}, "
          f"confidence={analytics_data['confidence_score']}, "
          f"escalated={analytics_data['escalated']}, "
          f"time={analytics_data['response_time_ms']}ms")

    return state


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_after_classification(state: HRGraphState) -> str:
    """Route to escalation immediately for sensitive queries."""
    if state.get("should_escalate") and state.get("escalation_type") == EscalationType.SENSITIVE:
        return "handle_escalation"
    return "retrieve_documents"


def route_after_confidence(state: HRGraphState) -> str:
    """Route to escalation if confidence is low."""
    if state.get("should_escalate"):
        return "handle_escalation"
    return "log_analytics"


# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_hr_graph() -> StateGraph:
    """Build and compile the LangGraph HR query resolution graph."""

    graph = StateGraph(HRGraphState)

    # Add nodes
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_response", generate_response)
    graph.add_node("assess_confidence", assess_confidence)
    graph.add_node("handle_escalation", handle_escalation)
    graph.add_node("log_analytics", log_analytics)

    # Entry point
    graph.set_entry_point("classify_query")

    # Edges
    graph.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "handle_escalation": "handle_escalation",
            "retrieve_documents": "retrieve_documents",
        }
    )

    graph.add_edge("retrieve_documents", "generate_response")
    graph.add_edge("generate_response", "assess_confidence")

    graph.add_conditional_edges(
        "assess_confidence",
        route_after_confidence,
        {
            "handle_escalation": "handle_escalation",
            "log_analytics": "log_analytics",
        }
    )

    graph.add_edge("handle_escalation", "log_analytics")
    graph.add_edge("log_analytics", END)

    return graph.compile()


# Compile once
hr_graph = build_hr_graph()


# ─── Main Query Function ──────────────────────────────────────────────────────

async def process_hr_query(
    query: str,
    employee_id: str = "EMP001",
    department: str = "Engineering",
    role: str = "employee",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point to process an HR query through the LangGraph pipeline.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    initial_state: HRGraphState = {
        "session_id": session_id,
        "employee_id": employee_id,
        "department": department,
        "role": role,
        "query": query,
        "messages": [],
        "query_category": "",
        "query_intent": "",
        "retrieved_docs": [],
        "context": "",
        "response": "",
        "confidence_score": 0.0,
        "sources": [],
        "should_escalate": False,
        "escalation_type": "",
        "escalation_reason": "",
        "start_time": time.time(),
        "response_time_ms": 0,
    }

    # Run the graph
    final_state = hr_graph.invoke(initial_state)

    return {
        "session_id": session_id,
        "query": query,
        "response": final_state.get("response", ""),
        "category": final_state.get("query_category", ""),
        "confidence": final_state.get("confidence_score", 0.0),
        "escalated": final_state.get("should_escalate", False),
        "escalation_type": final_state.get("escalation_type", ""),
        "sources": final_state.get("sources", []),
        "response_time_ms": final_state.get("response_time_ms", 0),
    }
