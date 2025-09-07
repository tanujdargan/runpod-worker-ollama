"""
ICD diagnostic coding agent using MedGemma-27B
Ported from langchain/pear-care/agents.py
"""

import json
import re
from typing import Dict, List, Optional
from ...clients.ollama_client import OllamaClient

class ICDAgent:
    """MedGemma-27B powered ICD diagnostic coding agent"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client
    
    async def warmup(self):
        """Warm up MedGemma"""
        try:
            await self.client.generate(
                prompt="test",
                model="medgemma:27b",
                max_tokens=1
            )
        except Exception:
            pass
    
    async def close(self):
        """Close client connections"""
        await self.client.cleanup()
    
    async def process(self, symptoms: Dict, questions_answers: List = None) -> Dict:
        """
        Generate ICD-10 codes for symptoms
        
        Args:
            symptoms: Dictionary with symptoms and pregnancy info
            questions_answers: Previous Q&A history
            
        Returns:
            Dictionary with ICD codes, importance ranking, and optional questions
        """
        qa_text = ""
        if questions_answers:
            qa_text = "Previous Q&A: " + "; ".join([
                f"Q: {qa['question']} A: {qa['answer']}" 
                for qa in questions_answers
            ])
        
        prompt = f"""You are a medical coder. Generate ICD-10 codes for these symptoms.

SYMPTOMS: {json.dumps(symptoms)}
{qa_text}

You may ask 1 clarifying question if critically needed.

Return valid JSON with importance ranking (1=most important, lower numbers = higher priority):
{{
    "icd_codes": [
        {{"code": "O14.0", "description": "Mild preeclampsia", "importance": 1}},
        {{"code": "R51", "description": "Headache", "importance": 2}}
    ],
    "questions": ["Question if needed"]
}}"""

        response = await self.client.generate(
            prompt=prompt,
            model="medgemma:27b",
            temperature=0.0,
            max_tokens=1000
        )
        
        result = self._parse_json(response)
        return result
    
    def _parse_json(self, response: str) -> Dict:
        """
        Parse JSON from MedGemma response
        
        Args:
            response: Raw response text from MedGemma
            
        Returns:
            Parsed dictionary or structured fallback
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from response using regex
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback: create basic structure from response
            return {
                "icd_codes": self._extract_icd_codes_fallback(response),
                "questions": self._extract_questions_fallback(response)
            }
    
    def _extract_icd_codes_fallback(self, response: str) -> List[Dict]:
        """Extract ICD codes when JSON parsing fails"""
        codes = []
        
        # Look for ICD-10 code patterns (letter followed by numbers)
        icd_pattern = r'([A-Z]\d{2}(?:\.\d{1,3})?)'
        matches = re.findall(icd_pattern, response)
        
        for i, code in enumerate(matches):
            # Try to extract description from context
            desc_pattern = rf'{re.escape(code)}[:\s]*([^,\n\.]+)'
            desc_match = re.search(desc_pattern, response, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else "Medical condition"
            
            codes.append({
                "code": code,
                "description": description,
                "importance": i + 1
            })
        
        # If no codes found, provide a generic one
        if not codes:
            codes.append({
                "code": "R68.89",
                "description": "Other general symptoms and signs",
                "importance": 1
            })
        
        return codes[:5]  # Limit to 5 codes
    
    def _extract_questions_fallback(self, response: str) -> List[str]:
        """Extract questions when JSON parsing fails"""
        questions = []
        
        # Look for question patterns
        question_patterns = [
            r'Question[:\s]*([^?\n]+\?)',
            r'(?:Can you|Could you|Do you|Are you|Have you|How)[^?\n]+\?',
            r'\?[^?]*\?'  # Text between question marks
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                question = match.strip()
                if len(question) > 10 and question not in questions:
                    questions.append(question)
        
        return questions[:1]  # Only return first question
