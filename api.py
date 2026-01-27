"""
AssistFlow AI - FastAPI Backend

REST API for the ticket analysis system.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
sys.path.append(".")

from src.pipeline import AssistFlowPipeline, TicketAnalysisResult
from src.ingestion import load_and_prepare_data, parse_resolution_time
from src.business_rules import process_business_rules, SLAResult, EscalationResult
from src.handler_decision import determine_handler, HandlerDecision
from src.llm_assistance import generate_llm_assistance
from config import DATA_PATH, COL_TICKET_ID, VALID_PRIORITIES, SLA_HOURS

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================
app = FastAPI(
    title="AssistFlow AI API",
    description="SLA-Aware Intelligent Customer Support System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AssistFlowPipeline] = None


# =============================================================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# =============================================================================
class TicketInput(BaseModel):
    """Input schema for ticket analysis."""
    ticket_id: Optional[str] = Field(None, description="Unique ticket identifier")
    subject: str = Field(..., description="Ticket subject line")
    description: str = Field(..., description="Ticket description")
    resolution_hours: Optional[float] = Field(
        None, 
        description="Historical resolution time in hours (for SLA simulation)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TICKET-001",
                "subject": "Product not working",
                "description": "My laptop screen is flickering and won't stop. I need urgent help.",
                "resolution_hours": 20.0
            }
        }


class TicketAnalysisResponse(BaseModel):
    """Response schema for ticket analysis."""
    ticket_id: Any
    full_text: str
    
    # ML Predictions
    predicted_priority: str
    priority_confidence: float
    issue_type: Optional[str]
    issue_type_confidence: Optional[float]
    
    # Business Rules
    sla_hours: int
    sla_status: str
    was_escalated: bool
    escalation_reason: Optional[str]
    final_priority: str
    
    # Handler Decision
    handler_type: str
    handler_reason: str
    
    # LLM Assistance
    ticket_summary: str
    explanation_text: str
    suggested_response: str


class BatchTicketInput(BaseModel):
    """Input schema for batch analysis."""
    tickets: List[TicketInput]


class BatchAnalysisResponse(BaseModel):
    """Response schema for batch analysis."""
    total_tickets: int
    results: List[TicketAnalysisResponse]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    timestamp: str


class SLAConfigResponse(BaseModel):
    """SLA configuration response."""
    sla_hours: Dict[str, int]
    valid_priorities: List[str]


# =============================================================================
# STARTUP EVENT
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Load ML models on startup."""
    global pipeline
    pipeline = AssistFlowPipeline()
    success = pipeline.load_models()
    if not success:
        print("WARNING: Models not loaded. Run train_models.py first.")


# =============================================================================
# API ENDPOINTS
# =============================================================================

# Health Check
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        models_loaded=pipeline is not None and pipeline._models_loaded,
        timestamp=datetime.now().isoformat()
    )


# Configuration
@app.get("/config/sla", response_model=SLAConfigResponse, tags=["Configuration"])
async def get_sla_config():
    """Get SLA configuration settings."""
    return SLAConfigResponse(
        sla_hours=SLA_HOURS,
        valid_priorities=VALID_PRIORITIES
    )


# Single Ticket Analysis
@app.post("/analyze", response_model=TicketAnalysisResponse, tags=["Analysis"])
async def analyze_ticket(ticket: TicketInput):
    """
    Analyze a single support ticket.
    
    This endpoint processes a ticket through the complete AssistFlow AI pipeline:
    1. ML-based priority prediction
    2. ML-based issue type prediction
    3. Rule-based SLA assignment
    4. Rule-based escalation logic
    5. Rule-based handler decision
    6. LLM-assisted response generation
    """
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run train_models.py first."
        )
    
    # Combine subject and description
    full_text = f"{ticket.subject} | {ticket.description}"
    
    # Run analysis
    result = pipeline.analyze_ticket(
        full_text=full_text,
        ticket_id=ticket.ticket_id or f"API-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        time_to_resolution_hours=ticket.resolution_hours
    )
    
    return TicketAnalysisResponse(**result.to_dict())


# Batch Analysis
@app.post("/analyze/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def analyze_batch(batch: BatchTicketInput):
    """
    Analyze multiple tickets in batch.
    
    Returns individual results plus summary statistics.
    """
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run train_models.py first."
        )
    
    results = []
    for i, ticket in enumerate(batch.tickets):
        full_text = f"{ticket.subject} | {ticket.description}"
        
        result = pipeline.analyze_ticket(
            full_text=full_text,
            ticket_id=ticket.ticket_id or f"BATCH-{i+1}",
            time_to_resolution_hours=ticket.resolution_hours
        )
        results.append(TicketAnalysisResponse(**result.to_dict()))
    
    # Calculate summary statistics
    summary = {
        "priority_distribution": {},
        "handler_distribution": {"Human": 0, "AI": 0},
        "sla_status_distribution": {"met": 0, "at_risk": 0, "breached": 0},
        "escalation_count": 0
    }
    
    for r in results:
        # Priority distribution
        summary["priority_distribution"][r.final_priority] = \
            summary["priority_distribution"].get(r.final_priority, 0) + 1
        
        # Handler distribution
        summary["handler_distribution"][r.handler_type] += 1
        
        # SLA status distribution
        summary["sla_status_distribution"][r.sla_status] += 1
        
        # Escalation count
        if r.was_escalated:
            summary["escalation_count"] += 1
    
    return BatchAnalysisResponse(
        total_tickets=len(results),
        results=results,
        summary=summary
    )


# Priority Prediction Only
@app.post("/predict/priority", tags=["Prediction"])
async def predict_priority(ticket: TicketInput):
    """
    Get priority prediction only (without full analysis).
    
    Useful for quick triage without LLM processing.
    """
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    
    full_text = f"{ticket.subject} | {ticket.description}"
    priority, confidence = pipeline.priority_model.get_prediction_confidence(full_text.lower())
    
    return {
        "predicted_priority": priority,
        "confidence": confidence
    }


# Issue Type Prediction Only
@app.post("/predict/issue-type", tags=["Prediction"])
async def predict_issue_type(ticket: TicketInput):
    """
    Get issue type prediction only.
    """
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    
    if pipeline.issue_type_model.model is None:
        raise HTTPException(status_code=404, detail="Issue type model not available.")
    
    full_text = f"{ticket.subject} | {ticket.description}"
    issue_type, confidence = pipeline.issue_type_model.get_prediction_confidence(full_text.lower())
    
    return {
        "issue_type": issue_type,
        "confidence": confidence
    }


# SLA Calculation
@app.post("/calculate/sla", tags=["Business Rules"])
async def calculate_sla(
    priority: str,
    resolution_hours: Optional[float] = None
):
    """
    Calculate SLA for a given priority level.
    
    Returns SLA hours, status, and escalation information.
    """
    if priority not in VALID_PRIORITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority. Must be one of: {VALID_PRIORITIES}"
        )
    
    sla_result, escalation_result = process_business_rules(
        predicted_priority=priority,
        time_to_resolution_hours=resolution_hours
    )
    
    return {
        "sla_hours": sla_result.sla_hours,
        "sla_status": sla_result.sla_status,
        "original_priority": escalation_result.original_priority,
        "final_priority": escalation_result.final_priority,
        "was_escalated": escalation_result.was_escalated,
        "escalation_reason": escalation_result.escalation_reason
    }


# Handler Decision
@app.post("/decide/handler", tags=["Business Rules"])
async def decide_handler(
    priority: str,
    issue_type: Optional[str] = None
):
    """
    Determine handler type (AI or Human) based on rules.
    """
    if priority not in VALID_PRIORITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority. Must be one of: {VALID_PRIORITIES}"
        )
    
    decision = determine_handler(priority, issue_type)
    
    return {
        "handler_type": decision.handler_type,
        "reason": decision.reason,
        "priority_triggered": decision.priority_triggered,
        "issue_type_triggered": decision.issue_type_triggered
    }


# Dataset Info
@app.get("/dataset/info", tags=["Dataset"])
async def get_dataset_info():
    """Get information about the loaded dataset."""
    try:
        df = load_and_prepare_data(DATA_PATH)
        return {
            "total_tickets": len(df),
            "columns": list(df.columns),
            "priority_distribution": df["Ticket Priority"].value_counts().to_dict(),
            "issue_type_distribution": df["Ticket Type"].value_counts().to_dict() if "Ticket Type" in df.columns else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Sample Tickets
@app.get("/dataset/sample", tags=["Dataset"])
async def get_sample_tickets(limit: int = 10):
    """Get sample tickets from the dataset."""
    try:
        df = load_and_prepare_data(DATA_PATH)
        sample = df.head(limit)
        
        tickets = []
        for _, row in sample.iterrows():
            tickets.append({
                "ticket_id": row[COL_TICKET_ID],
                "subject": row.get("Ticket Subject", ""),
                "description": row.get("Ticket Description", "")[:200] + "...",
                "priority": row.get("Ticket Priority", ""),
                "type": row.get("Ticket Type", "")
            })
        
        return {"tickets": tickets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RUN WITH UVICORN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
