"""
CPT procedure coding agent using MedGemma-27B
Ported from langchain/pear-care/agents.py
"""

import json
import re
from typing import Dict, List, Optional
from ...clients.ollama_client import OllamaClient

class CPTAgent:
    """MedGemma-27B powered CPT procedure coding agent"""
    
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
    
    async def process(self, symptoms: Dict, icd_codes: List, questions_answers: List = None) -> Dict:
        """
        Generate CPT procedure codes based on symptoms and diagnosis codes
        
        Args:
            symptoms: Dictionary with symptoms and pregnancy info
            icd_codes: List of ICD diagnostic codes
            questions_answers: Previous Q&A history
            
        Returns:
            Dictionary with CPT codes, importance ranking, explanation, and optional questions
        """
        qa_text = ""
        if questions_answers:
            qa_text = "Previous Q&A: " + "; ".join([
                f"Q: {qa['question']} A: {qa['answer']}" 
                for qa in questions_answers
            ])
        
        prompt = f"""You are a medical coder. Generate 2-3 most relevant CPT procedure codes based on symptoms and diagnosis codes.

SYMPTOMS: {json.dumps(symptoms)}
DIAGNOSIS CODES: {json.dumps(icd_codes)}
{qa_text}

Return valid JSON with 2-3 CPT codes maximum, importance ranking (1=most important), and brief explanation:
{{
    "cpt_codes": [
        {{"code": "99214", "description": "Office visit", "importance": 1}},
        {{"code": "36415", "description": "Blood draw", "importance": 2}}
    ],
    "explanation": "Brief explanation of procedures recommended for these symptoms",
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
                "cpt_codes": self._extract_cpt_codes_fallback(response),
                "explanation": self._extract_explanation_fallback(response),
                "questions": self._extract_questions_fallback(response)
            }
    
    def _extract_cpt_codes_fallback(self, response: str) -> List[Dict]:
        """Extract CPT codes when JSON parsing fails"""
        codes = []
        
        # Look for CPT code patterns (5-digit numbers)
        cpt_pattern = r'(\b\d{5}\b)'
        matches = re.findall(cpt_pattern, response)
        
        for i, code in enumerate(matches):
            # Try to extract description from context
            desc_pattern = rf'{re.escape(code)}[:\s]*([^,\n\.]+)'
            desc_match = re.search(desc_pattern, response, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else "Medical procedure"
            
            codes.append({
                "code": code,
                "description": description,
                "importance": i + 1
            })
        
        # If no codes found, provide common ones based on context
        if not codes:
            codes = [
                {
                    "code": "99214",
                    "description": "Office visit for established patient",
                    "importance": 1
                },
                {
                    "code": "36415",
                    "description": "Routine venipuncture",
                    "importance": 2
                }
            ]
        
        return codes[:3]  # Limit to 3 codes
    
    def _extract_explanation_fallback(self, response: str) -> str:
        """Extract explanation when JSON parsing fails"""
        # Look for explanation-like sentences
        sentences = response.split('.')
        
        explanation_keywords = [
            'procedure', 'recommended', 'evaluation', 'assessment', 
            'monitoring', 'treatment', 'diagnosis', 'care'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                for keyword in explanation_keywords:
                    if keyword.lower() in sentence.lower():
                        return sentence + '.'
        
        return "Standard evaluation and monitoring procedures recommended based on presented symptoms."
    
    def _extract_questions_fallback(self, response: str) -> List[str]:
        """Extract questions when JSON parsing fails"""
        questions = []
        
        # Look for question patterns
        question_patterns = [
            r'Question[:\s]*([^?\n]+\?)',
            r'(?:Can you|Could you|Do you|Are you|Have you|How|What)[^?\n]+\?',
            r'\?[^?]*\?'  # Text between question marks
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                question = match.strip()
                if len(question) > 10 and question not in questions:
                    questions.append(question)
        
        return questions[:1]  # Only return first question
