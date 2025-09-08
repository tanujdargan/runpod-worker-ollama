"""
Symptom extraction agent using GPT-5-nano
Ported from langchain/pear-care/agents.py
"""

import json
from typing import Dict, List, Optional, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ...clients.openai_client import OpenAIClient

class Symptoms(BaseModel):
    """Symptom extraction model"""
    symptoms: List[str] = Field(description="List of symptoms")
    pregnancy_related: bool = Field(description="Pregnancy related")

class SymptomAgent:
    """GPT-5-nano powered symptom extraction agent"""
    
    def __init__(self, openai_client: OpenAIClient):
        self.client = openai_client
        self.parser = JsonOutputParser()
    
    async def warmup(self):
        """Warm up GPT-5-nano"""
        try:
            await self.client.generate_response(
                prompt="test",
                model="gpt-5-nano",
                max_tokens=1
            )
        except Exception:
            pass
    
    async def process(self, user_input: str) -> Dict:
        """
        Extract symptoms from patient input
        
        Args:
            user_input: Patient's description of symptoms
            
        Returns:
            Dictionary with symptoms array and pregnancy_related boolean
        """
        format_instructions = self.parser.get_format_instructions()
        prompt = f"""Extract symptoms from patient input. Return JSON with symptoms array and pregnancy_related boolean. 
        
{format_instructions}

Input: {user_input}"""
        
        response = await self.client.generate_response(
            prompt=prompt,
            model="gpt-5-nano",
            max_tokens=500,
            temperature=0.1
        )
        
        return self.parser.parse(response)
    
    async def stream_process(self, user_input: str) -> AsyncGenerator[Dict, None]:
        """
        Stream symptom extraction results
        
        Args:
            user_input: Patient's description of symptoms
            
        Yields:
            Progressive symptom extraction results
        """
        format_instructions = self.parser.get_format_instructions()
        prompt = f"""Extract symptoms from patient input. Return JSON with symptoms array and pregnancy_related boolean.
        
{format_instructions}

Input: {user_input}"""
        
        # For streaming, we'll get the complete response and simulate progressive extraction
        response = await self.client.generate_response(
            prompt=prompt,
            model="gpt-5-nano",
            max_tokens=500,
            temperature=0.1
        )
        
        try:
            result = self.parser.parse(response)
            
            # Simulate streaming by yielding progressive JSON construction
            if result.get('symptoms'):
                for i, symptom in enumerate(result['symptoms']):
                    partial_symptoms = result['symptoms'][:i+1]
                    yield {
                        "symptoms": partial_symptoms,
                        "pregnancy_related": result.get('pregnancy_related', False)
                    }
                
                # Final complete result
                yield result
            else:
                yield result
                
        except Exception as e:
            # If parsing fails, return a basic structure
            yield {
                "symptoms": [user_input],
                "pregnancy_related": False,
                "parse_error": str(e)
            }
