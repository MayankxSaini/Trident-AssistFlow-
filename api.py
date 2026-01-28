"""AssistFlow AI - FastAPI Backend"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
sys.path.append(".")

from src.pipeline import AssistFlowPipeline
from src.ingestion import load_and_prepare_data
from src.business_rules import process_business_rules
from src.handler_decision import determine_handler
from config import DATA_PATH, COL_TICKET_ID, VALID_PRIORITIES, SLA_HOURS

app = FastAPI(
    title="AssistFlow AI API",
    description="Intelligent Customer Support System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[AssistFlowPipeline] = None


class TicketInput(BaseModel):
    ticket_id: Optional[str] = None
    subject: str
    description: str
    resolution_hours: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TICKET-001",
                "subject": "Product not working",
                "description": "My laptop screen is flickering.",
                "resolution_hours": 20.0
            }
        }


class TicketAnalysisResponse(BaseModel):
    ticket_id: Any
    full_text: str
    predicted_priority: str
    priority_confidence: float
    issue_type: Optional[str]
    issue_type_confidence: Optional[float]
    sla_hours: int
    sla_status: str
    was_escalated: bool
    escalation_reason: Optional[str]
    final_priority: str
    handler_type: str
    handler_reason: str
    ticket_summary: str
    explanation_text: str
    suggested_response: str


class BatchTicketInput(BaseModel):
    tickets: List[TicketInput]


class BatchAnalysisResponse(BaseModel):
    total_tickets: int
    results: List[TicketAnalysisResponse]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    timestamp: str


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = AssistFlowPipeline()
    success = pipeline.load_models()
    if not success:
        print("WARNING: Models not loaded. Run train_models.py first.")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=pipeline is not None and pipeline._models_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/config/sla", tags=["Configuration"])
async def get_sla_config():
    return {"sla_hours": SLA_HOURS, "valid_priorities": VALID_PRIORITIES}


@app.post("/analyze", response_model=TicketAnalysisResponse, tags=["Analysis"])
async def analyze_ticket(ticket: TicketInput):
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    full_text = f"{ticket.subject} | {ticket.description}"
    result = pipeline.analyze_ticket(
        full_text=full_text,
        ticket_id=ticket.ticket_id or f"API-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        time_to_resolution_hours=ticket.resolution_hours
    )
    return TicketAnalysisResponse(**result.to_dict())


@app.post("/analyze/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def analyze_batch(batch: BatchTicketInput):
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    for i, ticket in enumerate(batch.tickets):
        full_text = f"{ticket.subject} | {ticket.description}"
        result = pipeline.analyze_ticket(
            full_text=full_text,
            ticket_id=ticket.ticket_id or f"BATCH-{i+1}",
            time_to_resolution_hours=ticket.resolution_hours
        )
        results.append(TicketAnalysisResponse(**result.to_dict()))
    
    summary = {
        "priority_distribution": {},
        "handler_distribution": {"Human": 0, "AI": 0},
        "sla_status_distribution": {"met": 0, "at_risk": 0, "breached": 0},
        "escalation_count": 0
    }
    
    for r in results:
        summary["priority_distribution"][r.final_priority] = \
            summary["priority_distribution"].get(r.final_priority, 0) + 1
        summary["handler_distribution"][r.handler_type] += 1
        summary["sla_status_distribution"][r.sla_status] += 1
        if r.was_escalated:
            summary["escalation_count"] += 1
    
    return BatchAnalysisResponse(total_tickets=len(results), results=results, summary=summary)


@app.post("/predict/priority", tags=["Prediction"])
async def predict_priority(ticket: TicketInput):
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    full_text = f"{ticket.subject} | {ticket.description}"
    priority, confidence = pipeline.priority_model.get_prediction_confidence(full_text.lower())
    return {"predicted_priority": priority, "confidence": confidence}


@app.post("/predict/issue-type", tags=["Prediction"])
async def predict_issue_type(ticket: TicketInput):
    if pipeline is None or not pipeline._models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if pipeline.issue_type_model.model is None:
        raise HTTPException(status_code=404, detail="Issue type model not available")
    
    full_text = f"{ticket.subject} | {ticket.description}"
    issue_type, confidence = pipeline.issue_type_model.get_prediction_confidence(full_text.lower())
    return {"issue_type": issue_type, "confidence": confidence}


@app.post("/calculate/sla", tags=["Business Rules"])
async def calculate_sla(priority: str, resolution_hours: Optional[float] = None):
    if priority not in VALID_PRIORITIES:
        raise HTTPException(status_code=400, detail=f"Invalid priority. Use: {VALID_PRIORITIES}")
    
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


@app.post("/decide/handler", tags=["Business Rules"])
async def decide_handler(priority: str, issue_type: Optional[str] = None):
    if priority not in VALID_PRIORITIES:
        raise HTTPException(status_code=400, detail=f"Invalid priority. Use: {VALID_PRIORITIES}")
    
    decision = determine_handler(priority, issue_type)
    return {
        "handler_type": decision.handler_type,
        "reason": decision.reason,
        "priority_triggered": decision.priority_triggered,
        "issue_type_triggered": decision.issue_type_triggered
    }


@app.get("/dataset/info", tags=["Dataset"])
async def get_dataset_info():
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


@app.get("/dataset/sample", tags=["Dataset"])
async def get_sample_tickets(limit: int = 10):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
