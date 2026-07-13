import json
import logging
from typing import List
import httpx
from app.core.config import settings
from app.schemas.prompt import PromptAnalysisResponse

logger = logging.getLogger(__name__)

class PromptService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL

    async def analyze_prompt(self, prompt: str) -> PromptAnalysisResponse:
        if self.api_key:
            try:
                return await self._analyze_via_llm(prompt)
            except Exception as e:
                logger.warning(f"Gemini API prompt analysis failed: {str(e)}. Attempting local LLM.")
        
        try:
            return await self._analyze_via_local_llm(prompt)
        except Exception as e:
            logger.warning(f"Local LLM prompt analysis failed: {str(e)}. Falling back to local NLP parser.")
            
        return self._analyze_fallback(prompt)

    async def _analyze_via_local_llm(self, prompt: str) -> PromptAnalysisResponse:
        url = f"{settings.LOCAL_LLM_URL}/api/chat"
        
        system_instruction = (
            "You are an AI Dataset Factory assistant. You must analyze the natural language prompt describing desired images. "
            "You must return a structured JSON response matching this schema: "
            "{\n"
            '  "optimized_search_queries": ["query1", "query2"],\n'
            '  "synonyms": ["synonym1", "synonym2"],\n'
            '  "related_keywords": ["keyword1", "keyword2"],\n'
            '  "search_variations": ["variation1", "variation2"],\n'
            '  "detected_classes": ["class1", "class2"]\n'
            "}\n"
            "Ensure the response is a single, valid JSON object."
        )

        payload = {
            "model": settings.LOCAL_LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"{system_instruction}\n\nUser Input: {prompt}"
                }
            ],
            "format": "json",
            "stream": False
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            content = data["message"]["content"]
            parsed = json.loads(content)
            
            return PromptAnalysisResponse(
                original_prompt=prompt,
                optimized_search_queries=parsed.get("optimized_search_queries", []),
                synonyms=parsed.get("synonyms", []),
                related_keywords=parsed.get("related_keywords", []),
                search_variations=parsed.get("search_variations", []),
                detected_classes=parsed.get("detected_classes", [])
            )

    async def _analyze_via_llm(self, prompt: str) -> PromptAnalysisResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        system_instruction = (
            "You are an AI Dataset Factory assistant. You must analyze the natural language prompt describing desired images. "
            "You must return a structured JSON response matching this schema: "
            "{\n"
            '  "optimized_search_queries": ["query1", "query2"],\n'
            '  "synonyms": ["synonym1", "synonym2"],\n'
            '  "related_keywords": ["keyword1", "keyword2"],\n'
            '  "search_variations": ["variation1", "variation2"],\n'
            '  "detected_classes": ["class1", "class2"]\n'
            "}\n"
            "Ensure the response is a single, valid JSON object. Do not wrap it in markdown code blocks."
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_instruction}\n\nUser Input: {prompt}"}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            
            return PromptAnalysisResponse(
                original_prompt=prompt,
                optimized_search_queries=parsed.get("optimized_search_queries", []),
                synonyms=parsed.get("synonyms", []),
                related_keywords=parsed.get("related_keywords", []),
                search_variations=parsed.get("search_variations", []),
                detected_classes=parsed.get("detected_classes", [])
            )

    def _analyze_fallback(self, prompt: str) -> PromptAnalysisResponse:
        words = [w.strip(".,!?\"'").lower() for w in prompt.split()]
        stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "with", "for", "desiring", "desired", "of", "showing", "images"}
        
        filtered_words = [w for w in words if w and w not in stop_words]
        
        synonyms_map = {
            "car": ["automobile", "vehicle", "motorcar"],
            "cars": ["automobiles", "vehicles"],
            "dog": ["canine", "puppy", "hound"],
            "dogs": ["canines", "puppies"],
            "cat": ["feline", "kitten"],
            "cats": ["felines", "kittens"],
            "person": ["human", "individual"],
            "people": ["humans", "crowd"],
            "street": ["road", "avenue", "highway"],
            "highway": ["expressway", "freeway", "road"],
        }
        
        detected_classes = list(set(filtered_words))
        synonyms = []
        for word in detected_classes:
            if word in synonyms_map:
                synonyms.extend(synonyms_map[word])
                
        search_variations = [
            f"high quality photo of {prompt}",
            f"clear image of {prompt}",
            f"{prompt} dataset"
        ]
        
        optimized_search_queries = [
            " ".join(detected_classes),
            f"{' '.join(detected_classes)} photo"
        ]
        
        related_keywords = [f"related {word}" for word in detected_classes]

        return PromptAnalysisResponse(
            original_prompt=prompt,
            optimized_search_queries=optimized_search_queries,
            synonyms=list(set(synonyms)) if synonyms else ["alternate term"],
            related_keywords=related_keywords,
            search_variations=search_variations,
            detected_classes=detected_classes
        )
