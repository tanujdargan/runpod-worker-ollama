"""
Response models for the unified API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

class ChatChoice(BaseModel):
    """Single choice in chat completion"""
    index: int = Field(description="Choice index")
    message: Dict[str, str] = Field(description="Response message")
    finish_reason: Optional[str] = Field(description="Reason for completion")

class ChatUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = Field(description="Tokens in prompt")
    completion_tokens: int = Field(description="Tokens in completion")
    total_tokens: int = Field(description="Total tokens used")

class ChatResponse(BaseModel):
    """Chat completion response"""
    id: str = Field(description="Response ID")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    model: str = Field(description="Model used")
    choices: List[ChatChoice] = Field(description="Response choices")
    usage: Optional[ChatUsage] = Field(description="Token usage")

class StreamingChatChoice(BaseModel):
    """Streaming choice in chat completion"""
    index: int = Field(description="Choice index")
    delta: Dict[str, Any] = Field(description="Delta content")
    finish_reason: Optional[str] = Field(description="Reason for completion")

class StreamingChatResponse(BaseModel):
    """Streaming chat completion response"""
    id: str = Field(description="Response ID")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    model: str = Field(description="Model used")
    choices: List[StreamingChatChoice] = Field(description="Streaming choices")

class ConsultationStartResponse(BaseModel):
    """Response for starting consultation"""
    session_id: str = Field(description="Session identifier")
    status: str = Field(description="Processing status")
    current_stage: str = Field(description="Current workflow stage")

class ConsultationAnswerResponse(BaseModel):
    """Response for answering questions"""
    session_id: str = Field(description="Session identifier") 
    status: str = Field(description="Processing status")

class ICDCode(BaseModel):
    """ICD diagnostic code"""
    code: str = Field(description="ICD-10 code")
    description: str = Field(description="Code description")
    importance: int = Field(description="Importance ranking (1=most important)")

class CPTCode(BaseModel):
    """CPT procedure code"""
    code: str = Field(description="CPT code")
    description: str = Field(description="Procedure description")
    importance: int = Field(description="Importance ranking (1=most important)")

class DoctorRecommendation(BaseModel):
    """Doctor recommendation"""
    name: str = Field(description="Doctor/Department name")
    specialty: str = Field(description="Medical specialty")
    hospital: str = Field(description="Hospital/Institution")
    rating: float = Field(description="Rating score")
    match_score: Optional[float] = Field(description="Match score for symptoms")
    explanation: Optional[str] = Field(description="Why this doctor is recommended")
    location: Optional[str] = Field(description="Location/Address")
    contact: Optional[str] = Field(description="Contact information")

class ConsultationResults(BaseModel):
    """Complete consultation results"""
    symptoms: Optional[Dict] = Field(description="Extracted symptoms")
    icd_codes: Optional[List[ICDCode]] = Field(description="ICD diagnostic codes")
    cpt_codes: Optional[List[CPTCode]] = Field(description="CPT procedure codes")
    doctors: Optional[List[DoctorRecommendation]] = Field(description="Doctor recommendations")
    summary: Optional[str] = Field(description="Consultation summary")

class ConsultationResponse(BaseModel):
    """Consultation workflow response"""
    session_id: str = Field(description="Session identifier")
    status: str = Field(description="Processing status")
    results: ConsultationResults = Field(description="Consultation results")
    next_steps: Optional[List[str]] = Field(description="Recommended next steps")

class AgentUpdate(BaseModel):
    """Agent processing update"""
    agent: str = Field(description="Agent name")
    status: str = Field(description="Processing status")
    message: Optional[str] = Field(description="Status message")
    result: Optional[Dict] = Field(description="Agent result")
    questions: Optional[List[str]] = Field(description="Questions for user")
    timestamp: float = Field(description="Update timestamp")

class StreamingConsultationUpdate(BaseModel):
    """Streaming consultation update"""
    session_id: str = Field(description="Session identifier")
    agent: Optional[str] = Field(description="Current agent")
    type: str = Field(description="Update type: progress, streaming, question, final, complete")
    content: Any = Field(description="Update content")
    timestamp: float = Field(description="Update timestamp")

class ErrorResponse(BaseModel):
    """Error response"""
    error: Dict[str, str] = Field(description="Error details")
    session_id: Optional[str] = Field(description="Session identifier if applicable")
    timestamp: str = Field(description="Error timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Overall health status")
    timestamp: str = Field(description="Check timestamp")
    services: Dict[str, str] = Field(description="Individual service status")
    uptime: Optional[float] = Field(description="Service uptime in seconds")
    version: Optional[str] = Field(description="Service version")

class ModelsResponse(BaseModel):
    """Available models response"""
    data: List[Dict[str, Any]] = Field(description="List of available models")

class ModelInfo(BaseModel):
    """Individual model information"""
    id: str = Field(description="Model identifier")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    owned_by: Optional[str] = Field(description="Model owner")
    permission: Optional[List[Dict]] = Field(description="Model permissions")

class CompletionChoice(BaseModel):
    """Text completion choice"""
    text: str = Field(description="Generated text")
    index: int = Field(description="Choice index")
    finish_reason: Optional[str] = Field(description="Reason for completion")

class CompletionResponse(BaseModel):
    """Text completion response"""
    id: str = Field(description="Response ID")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    model: str = Field(description="Model used")
    choices: List[CompletionChoice] = Field(description="Completion choices")
    usage: Optional[ChatUsage] = Field(description="Token usage")

class MoreDoctorsResponse(BaseModel):
    """Additional doctors response"""
    session_id: str = Field(description="Session identifier")
    additional_doctors: List[DoctorRecommendation] = Field(description="Additional doctor recommendations")
    recommendations: Dict[str, Any] = Field(description="Additional recommendations")

# Utility response models
class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(default=True, description="Success status")
    message: str = Field(description="Success message")
    data: Optional[Any] = Field(description="Response data")

class StatusResponse(BaseModel):
    """Generic status response"""
    status: str = Field(description="Current status")
    message: Optional[str] = Field(description="Status message")
    details: Optional[Dict] = Field(description="Additional details")
