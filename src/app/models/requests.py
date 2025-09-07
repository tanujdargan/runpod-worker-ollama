"""
Request models for the unified API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(description="Message role: user, assistant, or system")
    content: str = Field(description="Message content")

class ChatRequest(BaseModel):
    """Chat completion request model"""
    model: str = Field(description="Model name (phraser, main, langchain, medgemma:27b)")
    messages: List[ChatMessage] = Field(description="List of chat messages")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    stream: bool = Field(default=False, description="Whether to stream the response")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")

class PatientData(BaseModel):
    """Patient demographic data"""
    age: Optional[int] = Field(default=None, description="Patient age")
    gender: Optional[str] = Field(default=None, description="Patient gender")
    weeks_pregnant: Optional[int] = Field(default=None, description="Weeks pregnant if applicable")
    pregnant: bool = Field(default=False, description="Whether patient is pregnant")
    location: Optional[str] = Field(default=None, description="Patient location")
    insurance: Optional[str] = Field(default=None, description="Insurance information")

class ChatHistory(BaseModel):
    """Chat history entry"""
    role: str = Field(description="Message role")
    content: str = Field(description="Message content")
    timestamp: str = Field(description="ISO timestamp")

class ConsultationRequest(BaseModel):
    """Langchain consultation request"""
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    patient_data: Optional[PatientData] = Field(default=None, description="Patient demographic data")
    symptoms: str = Field(description="Patient's symptom description")
    chat_history: Optional[List[ChatHistory]] = Field(default=[], description="Previous chat history")
    stream: bool = Field(default=True, description="Whether to stream responses")

class ConsultationAnswerRequest(BaseModel):
    """Answer to agent question"""
    session_id: str = Field(description="Session identifier")
    agent: str = Field(description="Agent that asked the question")
    answer: str = Field(description="User's answer")
    chat_history: Optional[List[ChatHistory]] = Field(default=[], description="Updated chat history")

class MoreDoctorsRequest(BaseModel):
    """Request for additional doctor recommendations"""
    session_id: str = Field(description="Session identifier")
    count: int = Field(default=5, description="Number of additional doctors to return")
    specialties: Optional[List[str]] = Field(default=None, description="Preferred specialties")
    location_preference: Optional[str] = Field(default=None, description="Location preference")

class CompletionRequest(BaseModel):
    """Text completion request (for direct model access)"""
    model: str = Field(description="Model name")
    prompt: str = Field(description="Input prompt")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    stream: bool = Field(default=False, description="Whether to stream the response")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")

class HealthCheckRequest(BaseModel):
    """Health check request"""
    detailed: bool = Field(default=False, description="Whether to return detailed health info")

# Validation schemas for specific model requests
class OpenAIRequest(ChatRequest):
    """Request specifically for OpenAI models"""
    model: str = Field(..., pattern="^(phraser|main|gpt-.*)")

class OllamaRequest(ChatRequest):
    """Request specifically for Ollama models"""
    model: str = Field(..., pattern="^(medgemma.*|langchain)")

class LangchainRequest(ConsultationRequest):
    """Request specifically for Langchain consultation"""
    pass
