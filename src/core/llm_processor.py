"""
LLM Processor untuk Mochi Bot
Menggunakan llama-cpp-python dengan Qwen2.5
"""

import json
import re
from typing import Dict, List, Any, Optional
from llama_cpp import Llama


# Pattern untuk mendeteksi function call yang salah format di content
FUNCTION_PATTERNS = [
    # Pattern: functions.function_name atau function_name:
    r'functions?\.(alert_caregiver|request_service|request_assistance)',
    r'^(alert_caregiver|request_service|request_assistance)\s*[:\(]',
    # Pattern: <function=name> atau <tool_call>
    r'<function[=\s]*(alert_caregiver|request_service|request_assistance)',
    r'<tool_call>.*?(alert_caregiver|request_service|request_assistance)',
]


class MochiLLM:
    """LLM Processor menggunakan llama. cpp"""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self. verbose = verbose
        self.llm:  Optional[Llama] = None
    
    def load(self) -> "MochiLLM": 
        """Load model ke memory"""
        print(f"   📂 Loading:  {self.model_path}")
        print(f"   🎮 GPU Layers: {self.n_gpu_layers}")
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            chat_format="chatml-function-calling",
            verbose=self. verbose
        )
        
        print("   ✅ Model loaded successfully!")
        return self
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 256
    ) -> Dict[str, Any]:
        """Generate response dari LLM"""
        
        if self.llm is None:
            raise RuntimeError("Model belum di-load!")
        
        # Simpan user message terakhir untuk context parsing
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break
        
        # Call LLM
        response = self. llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        message = response["choices"][0]["message"]
        
        result = {
            "content": message.get("content", ""),
            "function_call":  None
        }
        
        # Log untuk debugging
        print(f"🧠 LLM Raw Output - Content: '{result['content'][:100]}...' " if len(result['content']) > 100 else f"🧠 LLM Raw Output - Content: '{result['content']}'")
        
        # Check for function call from tool_calls
        if "tool_calls" in message and message["tool_calls"]: 
            tool_call = message["tool_calls"][0]
            result["function_call"] = {
                "name": tool_call["function"]["name"],
                "arguments": json.loads(tool_call["function"]["arguments"])
            }
            print(f"✅ Proper tool_call detected: {result['function_call']['name']}")
        
        # FALLBACK: Parse function call dari content jika model salah format
        # Model kecil kadang mengeluarkan "functions.request_service:" sebagai teks
        if result["function_call"] is None and result["content"]:
            print(f"⚠️ No tool_call, checking for malformed function in content...")
            print(f"   User message context: '{user_message[:80]}...'" if len(user_message) > 80 else f"   User message context: '{user_message}'")
            parsed = self._parse_malformed_function_call(result["content"], user_message)
            if parsed:
                print(f"✅ Parsed malformed function: {parsed['name']} with args: {parsed['arguments']}")
                result["function_call"] = parsed
                result["content"] = ""  # Clear content karena ini sebenarnya function call
            else:
                print(f"ℹ️ No function call detected, treating as conversation")
        
        return result
    
    def _parse_malformed_function_call(self, content: str, user_message: str = "") -> Optional[Dict[str, Any]]:
        """
        Parse function call yang salah format dari content.
        Model kecil kadang mengeluarkan format seperti:
        - "functions.request_service:"
        - "request_service(service_type='drink')"
        - "<function=alert_caregiver>"
        
        Args:
            content: Output dari LLM yang mungkin malformed
            user_message: Pesan user asli untuk context detection
        """
        content_lower = content.lower().strip()
        
        # Cek apakah ada pattern function call yang salah
        detected_function = None
        for pattern in FUNCTION_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                detected_function = match.group(1) if match.lastindex else None
                break
        
        if not detected_function:
            return None
        
        # Gabungkan content dan user_message untuk context detection yang lebih akurat
        # Prioritaskan user_message karena itu adalah request asli dari user
        context = f"{user_message} {content_lower}".lower()
        
        print(f"🔍 Parsing malformed function: {detected_function}")
        print(f"🔍 Context for parsing: {context[:100]}...")
        
        # Tentukan arguments berdasarkan function dan context
        if detected_function == "alert_caregiver":
            priority = "emergency"
            situation = "Permintaan bantuan darurat"
            
            if any(w in context for w in ["jatuh", "sakit", "sesak", "nyeri", "darurat"]):
                priority = "emergency"
                situation = "Kondisi darurat"
            elif any(w in context for w in ["tolong", "bantuan"]):
                priority = "urgent"
                situation = "Membutuhkan bantuan segera"
            
            return {
                "name": "alert_caregiver",
                "arguments": {
                    "priority": priority,
                    "situation": situation
                }
            }
        
        elif detected_function == "request_service":
            # Deteksi service_type dari context (prioritas: user_message)
            service_type = "other"
            details = ""
            
            # Check untuk minuman
            if any(w in context for w in ["minum", "haus", "teh", "kopi", "air", "drink", "susu"]):
                service_type = "drink"
                # Coba ekstrak detail spesifik
                if "teh" in context:
                    details = "teh"
                elif "kopi" in context:
                    details = "kopi"
                elif "susu" in context:
                    details = "susu"
                elif "air" in context:
                    details = "air putih"
                else:
                    details = "minuman"
            
            # Check untuk makanan
            elif any(w in context for w in ["makan", "food", "lapar", "bubur", "nasi", "roti"]):
                service_type = "food"
                if "bubur" in context:
                    details = "bubur"
                elif "nasi" in context:
                    details = "nasi"
                elif "roti" in context:
                    details = "roti"
                else:
                    details = "makanan"
            
            # Check untuk snack
            elif any(w in context for w in ["snack", "cemilan", "kue", "biskuit"]):
                service_type = "snack"
                details = "cemilan"
            
            # Check untuk obat
            elif any(w in context for w in ["obat", "medication", "vitamin"]):
                service_type = "medication"
                details = "obat"
            
            print(f"📦 Request Service: type={service_type}, details={details}")
            
            return {
                "name": "request_service",
                "arguments": {
                    "service_type": service_type,
                    "details": details
                }
            }
        
        elif detected_function == "request_assistance":
            # Deteksi assistance_type dari context (prioritas: user_message)
            assist_type = "other"
            urgency = "normal"
            
            # Check untuk toilet
            if any(w in context for w in ["toilet", "wc", "buang air", "pipis", "bab", "kamar mandi", "kencing"]):
                assist_type = "toilet"
                urgency = "soon"
            
            # Check untuk mandi
            elif any(w in context for w in ["mandi", "shower", "cuci muka"]):
                assist_type = "shower"
                urgency = "normal"
            
            # Check untuk mobilitas
            elif any(w in context for w in ["jalan", "mobility", "berdiri", "duduk", "bangun", "pindah"]):
                assist_type = "mobility"
                urgency = "normal"
            
            # Check untuk berpakaian
            elif any(w in context for w in ["baju", "pakaian", "dressing", "ganti baju", "celana"]):
                assist_type = "dressing"
                urgency = "normal"
            
            # Check urgency
            if any(w in context for w in ["segera", "cepat", "urgent", "kebelet"]):
                urgency = "urgent"
            
            print(f"🤝 Request Assistance: type={assist_type}, urgency={urgency}")
            
            return {
                "name": "request_assistance",
                "arguments": {
                    "assistance_type": assist_type,
                    "urgency": urgency
                }
            }
        
        return None