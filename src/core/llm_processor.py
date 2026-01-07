"""
LLM Processor untuk Mochi Bot
Menggunakan llama-cpp-python dengan Qwen2.5
"""

import json
from typing import Dict, List, Any, Optional
from llama_cpp import Llama


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
        
        # Check for function call
        if "tool_calls" in message and message["tool_calls"]: 
            tool_call = message["tool_calls"][0]
            result["function_call"] = {
                "name": tool_call["function"]["name"],
                "arguments": json.loads(tool_call["function"]["arguments"])
            }
        
        return result