"""
Doctor recommendation agent using GPT-5-nano
Ported from langchain/pear-care/agents.py
"""

import asyncio
from typing import Dict, List, Optional, AsyncGenerator
from ...clients.openai_client import OpenAIClient

# Doctor database from original implementation
DOCTORS = [
    {"name": "Mass General Maternal-Fetal Medicine", "specialty": "Maternal-Fetal Medicine", "hospital": "Mass General", "rating": 4.9},
    {"name": "Brigham Women's Obstetrics", "specialty": "Obstetrics", "hospital": "Brigham Women's", "rating": 4.8},
    {"name": "Boston Medical Center Internal Medicine", "specialty": "Internal Medicine", "hospital": "Boston Medical", "rating": 4.7},
    {"name": "Newton-Wellesley Family Medicine", "specialty": "Family Medicine", "hospital": "Newton-Wellesley", "rating": 4.6},
    {"name": "Beth Israel Emergency Medicine", "specialty": "Emergency Medicine", "hospital": "Beth Israel", "rating": 4.5},
    {"name": "Tufts Medical Cardiology", "specialty": "Cardiology", "hospital": "Tufts Medical", "rating": 4.8},
    {"name": "Harvard Vanguard Primary Care", "specialty": "Primary Care", "hospital": "Harvard Vanguard", "rating": 4.6},
    {"name": "Children's Hospital Pediatrics", "specialty": "Pediatrics", "hospital": "Children's Hospital", "rating": 4.9},
    {"name": "McLean Hospital Psychiatry", "specialty": "Psychiatry", "hospital": "McLean Hospital", "rating": 4.7},
    {"name": "Spaulding Rehabilitation", "specialty": "Physical Medicine", "hospital": "Spaulding", "rating": 4.5}
]

class DoctorAgent:
    """GPT-5-nano powered doctor recommendation agent"""
    
    def __init__(self, openai_client: OpenAIClient):
        self.client = openai_client
    
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
    
    def select_doctors(self, icd_codes: List, count: int = 2) -> List[Dict]:
        """
        Select doctors based on ICD codes and specialty matching
        
        Args:
            icd_codes: List of ICD diagnostic codes
            count: Number of doctors to return
            
        Returns:
            List of selected doctors with match scores
        """
        # Extract specialties based on ICD codes
        specialties = []
        for code in icd_codes:
            desc = code.get('description', '').lower()
            code_str = code.get('code', '').upper()
            
            # Pregnancy/Obstetric conditions
            if any(keyword in desc for keyword in ['pregnancy', 'obstetric', 'preeclampsia', 'maternal']):
                specialties.extend(['Maternal-Fetal Medicine', 'Obstetrics'])
            
            # Cardiac conditions
            elif any(keyword in desc for keyword in ['cardiac', 'heart', 'cardio']):
                specialties.append('Cardiology')
            
            # Mental health conditions
            elif any(keyword in desc for keyword in ['depression', 'anxiety', 'mental', 'psychiatric']):
                specialties.append('Psychiatry')
            
            # Pediatric conditions (age-based or specific codes)
            elif code_str.startswith('P') or any(keyword in desc for keyword in ['pediatric', 'child', 'infant']):
                specialties.append('Pediatrics')
            
            # Emergency conditions
            elif any(keyword in desc for keyword in ['emergency', 'acute', 'trauma']):
                specialties.append('Emergency Medicine')
            
            # Musculoskeletal conditions
            elif any(keyword in desc for keyword in ['musculoskeletal', 'fracture', 'joint', 'muscle']):
                specialties.append('Physical Medicine')
            
            # Default to internal medicine and family medicine
            else:
                specialties.extend(['Internal Medicine', 'Family Medicine', 'Primary Care'])
        
        # Score doctors by specialty match and rating
        scored_doctors = []
        for doctor in DOCTORS:
            score = doctor['rating']  # Base score from rating
            
            # Add bonus for specialty match
            if doctor['specialty'] in specialties:
                score += 1.0
            
            # Add smaller bonus for general practitioners
            if doctor['specialty'] in ['Internal Medicine', 'Family Medicine', 'Primary Care']:
                score += 0.5
            
            scored_doctors.append({
                **doctor,
                'match_score': score
            })
        
        # Sort by match score (highest first) and return top N
        scored_doctors.sort(key=lambda x: x['match_score'], reverse=True)
        return scored_doctors[:count]
    
    async def generate_explanations(self, doctors: List[Dict], symptoms: Dict) -> List[Dict]:
        """
        Generate explanations for why each doctor is recommended
        
        Args:
            doctors: List of selected doctors
            symptoms: Patient symptoms dictionary
            
        Returns:
            List of doctors with explanations
        """
        results = []
        for doctor in doctors:
            explanation = await self._generate_single_explanation(doctor, symptoms)
            results.append({
                **doctor,
                'explanation': explanation
            })
        
        return results
    
    async def stream_generate_explanations(self, doctors: List[Dict], symptoms: Dict) -> AsyncGenerator[Dict, None]:
        """
        Stream doctor explanations as they're generated
        
        Args:
            doctors: List of selected doctors
            symptoms: Patient symptoms dictionary
            
        Yields:
            Progressive doctor explanations with streaming chunks
        """
        for doctor in doctors:
            explanation = await self._generate_single_explanation(doctor, symptoms)
            
            # Simulate streaming by yielding word-by-word chunks
            words = explanation.split()
            
            # Yield first few words as chunks
            for i, word in enumerate(words[:5]):
                yield {
                    **doctor,
                    'explanation_chunk': word + (' ' if i < len(words) - 1 else ''),
                    'is_final': False
                }
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.1)
            
            # Final result for this doctor
            yield {
                **doctor,
                'explanation': explanation,
                'is_final': True
            }
    
    async def _generate_single_explanation(self, doctor: Dict, symptoms: Dict) -> str:
        """Generate explanation for a single doctor recommendation"""
        symptom_list = symptoms.get('symptoms', [])
        
        prompt = f"""Explain why {doctor['name']} is the right choice for treating: {symptom_list}

Write about the department's expertise with these specific symptoms. Focus on:
- What procedures/tests they perform for these symptoms
- Their specialized training and experience  
- Why patients choose this department

Be direct and factual. No hedging language like "would" or "I cannot verify". 2 sentences maximum."""
        
        explanation = await self.client.generate_response(
            prompt=prompt,
            model="gpt-5-nano",
            max_tokens=150,
            temperature=0.3
        )
        
        return explanation.strip()
    
    async def get_more_doctors(self, icd_codes: List, current_doctors: List[str], count: int = 3) -> List[Dict]:
        """
        Get additional doctor recommendations
        
        Args:
            icd_codes: List of ICD diagnostic codes
            current_doctors: Names of already recommended doctors
            count: Number of additional doctors to return
            
        Returns:
            List of additional doctors with explanations
        """
        # Get all potential doctors
        all_doctors = self.select_doctors(icd_codes, count=len(DOCTORS))
        
        # Filter out already recommended doctors
        additional_doctors = [
            doctor for doctor in all_doctors 
            if doctor['name'] not in current_doctors
        ]
        
        # Return top N additional doctors
        selected_additional = additional_doctors[:count]
        
        # Generate explanations for additional doctors
        results = []
        for doctor in selected_additional:
            # Create a mock symptoms dict for explanation
            symptoms = {"symptoms": ["additional medical consultation"]}
            explanation = await self._generate_single_explanation(doctor, symptoms)
            results.append({
                **doctor,
                'explanation': explanation
            })
        
        return results
