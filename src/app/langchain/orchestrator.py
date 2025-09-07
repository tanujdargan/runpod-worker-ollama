"""
Langchain orchestrator ported from langchain/pear-care
Maintains the 5-stage pipeline: Intake → ICD Coding → CPT Coding → Provider Matching → Summary
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime

from ..clients.openai_client import OpenAIClient
from ..clients.ollama_client import OllamaClient
from .agents.symptom_agent import SymptomAgent
from .agents.icd_agent import ICDAgent
from .agents.cpt_agent import CPTAgent
from .agents.doctor_agent import DoctorAgent

class LangchainOrchestrator:
    """
    Main consultation workflow orchestrator
    Manages the 5-agent pipeline with streaming and progressive warmup
    """
    
    def __init__(self, openai_client: OpenAIClient, ollama_client: OllamaClient):
        self.openai_client = openai_client
        self.ollama_client = ollama_client
        
        # Initialize agents
        self.symptom_agent = SymptomAgent(openai_client)
        self.icd_agent = ICDAgent(ollama_client)
        self.cpt_agent = CPTAgent(ollama_client)
        self.doctor_agent = DoctorAgent(openai_client)
        
        # Session storage (in production, use Redis or database)
        self.sessions = {}
    
    async def warmup(self):
        """Warm up all agents for faster responses"""
        print("🔄 Warming up Langchain agents...")
        try:
            # Warm up all agents in parallel
            await asyncio.gather(
                self.symptom_agent.warmup(),
                self.icd_agent.warmup(),
                self.cpt_agent.warmup(),
                self.doctor_agent.warmup(),
                return_exceptions=True
            )
            print("✅ Langchain agents warmed up")
        except Exception as e:
            print(f"⚠️  Langchain warmup failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if all agents are healthy"""
        try:
            # Test each agent with a minimal request
            await self.symptom_agent.warmup()
            await self.icd_agent.warmup()
            await self.cpt_agent.warmup() 
            await self.doctor_agent.warmup()
            return True
        except Exception:
            return False
    
    async def run_consultation(
        self,
        user_input: str,
        patient_data: Optional[Dict] = None,
        session_id: Optional[str] = None,
        qa_history: List = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Main consultation workflow with Q&A support
        
        Args:
            user_input: Patient's symptom description
            patient_data: Patient demographic data
            session_id: Session identifier for persistence
            qa_history: Previous Q&A exchanges
            
        Yields:
            Streaming updates from each agent
        """
        agents_to_cleanup = []
        
        try:
            # Initialize session
            if session_id:
                self.sessions[session_id] = {
                    "created_at": datetime.now().isoformat(),
                    "patient_data": patient_data or {},
                    "symptoms": user_input,
                    "qa_history": qa_history or [],
                    "results": {}
                }
            
            # Step 1: Stream symptom extraction while warming up ICD in parallel
            yield {"agent": "symptom", "status": "processing", "message": "Extracting symptoms..."}
            
            # Start ICD agent warmup in background
            icd_warmup_task = asyncio.create_task(self.icd_agent.warmup())
            
            # Stream symptoms
            symptoms = None
            stream_started = False
            async for chunk in self.symptom_agent.stream_process(user_input):
                if not stream_started:
                    yield {"agent": "symptom", "status": "stream_started"}
                    stream_started = True
                yield {"agent": "symptom", "status": "streaming", "chunk": chunk}
                symptoms = chunk  # Keep final result
            
            yield {"agent": "symptom", "status": "complete", "result": symptoms}
            
            if session_id:
                self.sessions[session_id]["results"]["symptoms"] = symptoms
            
            # Ensure ICD warmup completes
            await icd_warmup_task
            
            # Step 2: ICD analysis while warming up CPT in parallel
            yield {"agent": "icd", "status": "processing", "message": "Analyzing diagnostic codes..."}
            
            # Start CPT agent warmup in background
            cpt_warmup_task = asyncio.create_task(self.cpt_agent.warmup())
            
            icd_result = await self.icd_agent.process(symptoms, qa_history)
            yield {"agent": "icd", "status": "complete", "result": icd_result}
            
            if session_id:
                self.sessions[session_id]["results"]["icd"] = icd_result
            
            # Ensure CPT warmup completes
            await cpt_warmup_task
            
            # Check for ICD questions
            if icd_result.get('questions'):
                yield {
                    "agent": "icd", 
                    "status": "question", 
                    "questions": icd_result['questions'],
                    "message": "Need additional information for diagnosis"
                }
                return
            
            # Step 3: CPT analysis while warming up doctor agent
            yield {"agent": "cpt", "status": "processing", "message": "Generating procedure codes..."}
            
            # Start doctor agent warmup in background
            doctor_warmup_task = asyncio.create_task(self.doctor_agent.warmup())
            
            # Start CPT processing
            cpt_task = asyncio.create_task(
                self.cpt_agent.process(
                    symptoms, 
                    icd_result.get('icd_codes', []), 
                    qa_history
                )
            )
            
            # Step 4: Doctor selection (prepare while CPT runs)
            yield {"agent": "doctor", "status": "processing", "message": "Selecting healthcare providers..."}
            selected_doctors = self.doctor_agent.select_doctors(
                icd_result.get('icd_codes', []), 
                count=2
            )
            
            # Ensure doctor warmup completes
            await doctor_warmup_task
            
            # Wait for CPT to complete first
            cpt_result = await cpt_task
            yield {"agent": "cpt", "status": "complete", "result": cpt_result}
            
            if session_id:
                self.sessions[session_id]["results"]["cpt"] = cpt_result
            
            # Check for CPT questions
            if cpt_result.get('questions'):
                yield {
                    "agent": "cpt", 
                    "status": "question", 
                    "questions": cpt_result['questions'],
                    "message": "Need additional information for procedures"
                }
                return
            
            # Stream doctor explanations
            final_doctors = []
            stream_started = False
            async for doctor_chunk in self.doctor_agent.stream_generate_explanations(selected_doctors, symptoms):
                if not stream_started:
                    yield {"agent": "doctor", "status": "stream_started"}
                    stream_started = True
                if doctor_chunk.get('is_final'):
                    final_doctors.append(doctor_chunk)
                    yield {"agent": "doctor", "status": "doctor_complete", "result": doctor_chunk}
                else:
                    yield {"agent": "doctor", "status": "streaming", "chunk": doctor_chunk}
            
            yield {"agent": "doctor", "status": "complete", "result": final_doctors}
            
            if session_id:
                self.sessions[session_id]["results"]["doctors"] = final_doctors
                self.sessions[session_id]["status"] = "completed"
            
            # Final summary
            yield {"status": "consultation_complete", "session_id": session_id}
            
        except Exception as e:
            yield {"status": "error", "error": str(e), "session_id": session_id}
            
        finally:
            # Cleanup sessions for agents that need it
            for agent in agents_to_cleanup:
                try:
                    await agent.close()
                except Exception:
                    pass
    
    async def continue_consultation_with_answer(
        self,
        session_id: str,
        question_agent: str,
        answer: str
    ) -> AsyncGenerator[Dict, None]:
        """
        Continue consultation after answering a question
        
        Args:
            session_id: Session identifier
            question_agent: Agent that asked the question
            answer: User's answer
            
        Yields:
            Streaming updates for remaining workflow
        """
        if session_id not in self.sessions:
            yield {"status": "error", "error": "Session not found"}
            return
        
        session_data = self.sessions[session_id]
        
        # Add Q&A to history
        qa_history = session_data.get("qa_history", [])
        last_question = session_data.get("results", {}).get(question_agent, {}).get("questions", [""])
        if last_question:
            qa_history.append({
                "question": last_question[0],
                "answer": answer,
                "agent": question_agent
            })
        
        session_data["qa_history"] = qa_history
        
        agents_to_cleanup = []
        
        try:
            if question_agent == "icd":
                # Re-run ICD with answer
                yield {"agent": "icd", "status": "processing", "message": "Re-analyzing with additional information..."}
                
                symptoms = session_data["results"].get("symptoms", {})
                icd_result = await self.icd_agent.process(symptoms, qa_history)
                yield {"agent": "icd", "status": "complete", "result": icd_result}
                
                session_data["results"]["icd"] = icd_result
                
                # Continue with CPT if no more questions
                if not icd_result.get('questions'):
                    yield {"agent": "cpt", "status": "processing", "message": "Generating procedure codes..."}
                    
                    cpt_result = await self.cpt_agent.process(
                        symptoms, 
                        icd_result.get('icd_codes', []), 
                        qa_history
                    )
                    yield {"agent": "cpt", "status": "complete", "result": cpt_result}
                    
                    session_data["results"]["cpt"] = cpt_result
                    
                    if not cpt_result.get('questions'):
                        # Complete with doctors
                        yield {"agent": "doctor", "status": "processing", "message": "Selecting healthcare providers..."}
                        
                        selected_doctors = self.doctor_agent.select_doctors(
                            icd_result.get('icd_codes', []), 
                            count=2
                        )
                        doctors_with_explanations = await self.doctor_agent.generate_explanations(
                            selected_doctors, 
                            symptoms
                        )
                        yield {"agent": "doctor", "status": "complete", "result": doctors_with_explanations}
                        
                        session_data["results"]["doctors"] = doctors_with_explanations
                        session_data["status"] = "completed"
                        
                        yield {"status": "consultation_complete", "session_id": session_id}
            
            elif question_agent == "cpt":
                # Re-run CPT with answer
                yield {"agent": "cpt", "status": "processing", "message": "Re-analyzing procedures with additional information..."}
                
                symptoms = session_data["results"].get("symptoms", {})
                icd_codes = session_data["results"].get("icd", {}).get("icd_codes", [])
                cpt_result = await self.cpt_agent.process(symptoms, icd_codes, qa_history)
                yield {"agent": "cpt", "status": "complete", "result": cpt_result}
                
                session_data["results"]["cpt"] = cpt_result
                
                if not cpt_result.get('questions'):
                    # Complete with doctors
                    yield {"agent": "doctor", "status": "processing", "message": "Selecting healthcare providers..."}
                    
                    selected_doctors = self.doctor_agent.select_doctors(icd_codes, count=2)
                    doctors_with_explanations = await self.doctor_agent.generate_explanations(
                        selected_doctors, 
                        symptoms
                    )
                    yield {"agent": "doctor", "status": "complete", "result": doctors_with_explanations}
                    
                    session_data["results"]["doctors"] = doctors_with_explanations
                    session_data["status"] = "completed"
                    
                    yield {"status": "consultation_complete", "session_id": session_id}
            
        except Exception as e:
            yield {"status": "error", "error": str(e), "session_id": session_id}
            
        finally:
            for agent in agents_to_cleanup:
                try:
                    await agent.close()
                except Exception:
                    pass
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    async def cleanup(self):
        """Cleanup orchestrator resources"""
        # Cleanup would happen automatically with agent cleanup
        # In production, this might close database connections, etc.
        pass
