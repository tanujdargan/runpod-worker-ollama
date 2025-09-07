"""
Langchain consultation router for medical consultation workflow
"""

import json
import time
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from ..langchain.orchestrator import LangchainOrchestrator
from ..services.streaming import StreamingManager
from ..models.requests import ConsultationRequest, ConsultationAnswerRequest, MoreDoctorsRequest
from ..models.responses import (
    ConsultationStartResponse, 
    ConsultationAnswerResponse, 
    MoreDoctorsResponse,
    ErrorResponse
)

router = APIRouter()

# These will be injected by the main app
langchain_orchestrator: LangchainOrchestrator = None
streaming_manager: StreamingManager = None

def set_dependencies(lo: LangchainOrchestrator, sm: StreamingManager):
    """Set dependencies (called from main app)"""
    global langchain_orchestrator, streaming_manager
    langchain_orchestrator = lo
    streaming_manager = sm

@router.post("/langchain/consultation", response_model=ConsultationStartResponse)
async def start_consultation(
    request: ConsultationRequest,
    auth_result = Depends(lambda: None)  # Will be overridden by main app
):
    """
    Start Langchain consultation workflow
    
    Initiates the 5-agent pipeline:
    1. Symptom extraction (GPT-5-nano)
    2. ICD diagnostic coding (MedGemma-27B)
    3. CPT procedure coding (MedGemma-27B)
    4. Provider matching
    5. Summary generation
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Validate input
        if not request.symptoms or not request.symptoms.strip():
            raise HTTPException(
                status_code=400,
                detail="Symptoms description is required"
            )
        
        # Store consultation request in orchestrator
        patient_data = request.patient_data.dict() if request.patient_data else {}
        chat_history = [h.dict() for h in request.chat_history] if request.chat_history else []
        
        return ConsultationStartResponse(
            session_id=session_id,
            status="processing",
            current_stage="symptom_extraction"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/langchain/consultation/{session_id}/stream")
async def stream_consultation(session_id: str, auth_result = Depends(lambda: None)):
    """
    Stream consultation updates (Server-Sent Events)
    
    Returns real-time updates as agents process the consultation:
    - Agent progress updates
    - Streaming results as agents complete
    - Questions when agents need clarification
    - Final completion notification
    """
    try:
        # Get session data from orchestrator
        session_data = langchain_orchestrator.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        user_input = session_data.get("symptoms", "")
        patient_data = session_data.get("patient_data", {})
        qa_history = session_data.get("qa_history", [])
        
        async def generate_updates():
            try:
                async for update in langchain_orchestrator.run_consultation(
                    user_input=user_input,
                    patient_data=patient_data,
                    session_id=session_id,
                    qa_history=qa_history
                ):
                    # Transform internal format to API format
                    if update.get('agent'):
                        agent = update['agent']
                        status = update.get('status')
                        
                        if status == 'processing':
                            data = {
                                "agent": agent,
                                "type": "progress",
                                "content": update.get('message', f"{agent.title()} agent processing...")
                            }
                        elif status == 'stream_started':
                            data = {
                                "agent": agent,
                                "type": "stream_started",
                                "content": f"{agent.title()} streaming started"
                            }
                        elif status == 'streaming':
                            data = {
                                "agent": agent,
                                "type": "streaming",
                                "content": update.get('chunk', {})
                            }
                        elif status == 'complete':
                            data = {
                                "agent": agent,
                                "type": "final",
                                "content": update.get('result', {})
                            }
                        elif status == 'doctor_complete':
                            data = {
                                "agent": "doctor",
                                "type": "doctor_final",
                                "content": update.get('result', {})
                            }
                        elif status == 'question':
                            data = {
                                "agent": agent,
                                "type": "question",
                                "content": {
                                    "question": update.get('questions', [''])[0],
                                    "options": [],
                                    "message": update.get('message', '')
                                }
                            }
                        else:
                            data = update
                    elif update.get('status') == 'consultation_complete':
                        data = {
                            "type": "complete",
                            "session_id": session_id,
                            "message": "Consultation completed successfully"
                        }
                    elif update.get('status') == 'error':
                        data = {
                            "type": "error",
                            "error": {
                                "message": update.get('error', 'Unknown error'),
                                "type": "consultation_error"
                            },
                            "session_id": session_id
                        }
                    else:
                        data = update
                    
                    yield f"data: {json.dumps(data)}\n\n"
                
            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": {
                        "message": str(e),
                        "type": "streaming_error"
                    },
                    "session_id": session_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_updates(),
            media_type="text/event-stream",
            headers=streaming_manager.create_sse_headers()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langchain/answer", response_model=ConsultationAnswerResponse)
async def answer_question(
    request: ConsultationAnswerRequest,
    auth_result = Depends(lambda: None)
):
    """
    Answer agent question and resume consultation
    
    When an agent asks a clarifying question, this endpoint
    provides the answer and automatically resumes the workflow.
    """
    try:
        # Validate session exists
        session_data = langchain_orchestrator.get_session(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate answer
        if not request.answer or not request.answer.strip():
            raise HTTPException(
                status_code=400,
                detail="Answer is required"
            )
        
        # The orchestrator will handle continuing the consultation
        # The stream endpoint will automatically resume after this call
        
        return ConsultationAnswerResponse(
            session_id=request.session_id,
            status="processing"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langchain/more-doctors", response_model=MoreDoctorsResponse)
async def get_more_doctors(
    request: MoreDoctorsRequest,
    auth_result = Depends(lambda: None)
):
    """
    Get additional doctor recommendations
    
    Returns additional healthcare providers based on the 
    consultation results and user preferences.
    """
    try:
        # Validate session exists
        session_data = langchain_orchestrator.get_session(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get consultation results
        results = session_data.get("results", {})
        icd_codes = results.get("icd", {}).get("icd_codes", [])
        current_doctors = []
        
        if "doctors" in results:
            current_doctors = [doc.get("name", "") for doc in results["doctors"]]
        
        # Get additional doctors from the doctor agent
        from ..langchain.agents.doctor_agent import DoctorAgent
        from ..clients.openai_client import OpenAIClient
        
        # Use existing client (this is a temporary solution)
        # In production, we'd get this from the orchestrator
        openai_client = OpenAIClient()
        doctor_agent = DoctorAgent(openai_client)
        
        additional_doctors = await doctor_agent.get_more_doctors(
            icd_codes=icd_codes,
            current_doctors=current_doctors,
            count=request.count
        )
        
        return MoreDoctorsResponse(
            session_id=request.session_id,
            additional_doctors=additional_doctors,
            recommendations={
                "summary": f"Found {len(additional_doctors)} additional specialists",
                "next_steps": [
                    "Schedule appointment with preferred provider",
                    "Prepare medical history for consultation",
                    "Consider second opinion if needed"
                ]
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/langchain/session/{session_id}")
async def get_session(session_id: str, auth_result = Depends(lambda: None)):
    """
    Get consultation session data
    
    Returns current state and results of the consultation session.
    """
    try:
        session_data = langchain_orchestrator.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "status": session_data.get("status", "unknown"),
            "created_at": session_data.get("created_at"),
            "patient_data": session_data.get("patient_data", {}),
            "symptoms": session_data.get("symptoms", ""),
            "results": session_data.get("results", {}),
            "qa_history": session_data.get("qa_history", [])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/langchain/session/{session_id}")
async def delete_session(session_id: str, auth_result = Depends(lambda: None)):
    """
    Delete consultation session
    
    Removes session data and cleans up resources.
    """
    try:
        success = langchain_orchestrator.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "status": "deleted",
            "message": "Session deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langchain/test")
async def test_langchain(auth_result = Depends(lambda: None)):
    """Test endpoint for Langchain functionality"""
    try:
        # Test basic consultation workflow
        test_session_id = str(uuid.uuid4())
        test_symptoms = "I have a severe headache and nausea"
        
        # Start a test consultation
        results = []
        async for update in langchain_orchestrator.run_consultation(
            user_input=test_symptoms,
            patient_data={"age": 30, "gender": "female"},
            session_id=test_session_id
        ):
            results.append(update)
            # Stop after first few updates for testing
            if len(results) >= 3:
                break
        
        return {
            "status": "success",
            "test_session_id": test_session_id,
            "test_symptoms": test_symptoms,
            "sample_updates": results,
            "timestamp": time.time()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }
