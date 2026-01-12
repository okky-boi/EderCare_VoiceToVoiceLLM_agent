"""
Mochi Bot - Main Application
Elderly Care AI Companion (Bahasa Indonesia)

Author: okky-boi
Created: 2026-01-06
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.llm_processor import MochiLLM
from core.tts_service import TTSService
from core.action_dispatcher import ActionDispatcher
from core.stt_service import STTService
from prompts.system_prompts import SYSTEM_PROMPT_ID, TOOLS_DEFINITION

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class MochiConfig:
    """Konfigurasi Mochi Bot"""
    # Paths - menggunakan relative paths untuk portability
    project_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    # Model ringan untuk CPU
    
    # User settings
    user_name: str = "Nenek Sari"
    preferred_name: str = "Nenek"
    caregiver_name: str = "Ibu Rina"
    
    # TTS settings
    tts_voice: str = "id-ID-GadisNeural"  # atau "id-ID-ArdiNeural" untuk pria
    tts_rate: str = "-5%"  # Sedikit lebih lambat
    
    # LLM settings - optimized untuk CPU
    n_ctx: int = 2048  # Dikurangi untuk CPU
    n_gpu_layers: int = 0  # 0 = CPU only
    temperature: float = 0.7
    max_tokens: int = 150  # Dikurangi untuk response lebih cepat
    
    # STT settings - optimized untuk CPU
    stt_model: str = "base"  # tiny, base, small, medium, large-v3
    stt_device: str = "cpu"  # cpu karena tidak ada GPU
    stt_compute_type: str = "int8"  # int8 lebih cepat di CPU
    
    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8051


# ============================================================
# RESPONSE TYPES
# ============================================================

class ResponseType(Enum):
    CONVERSATION = "conversation"
    ACTION = "action"


@dataclass
class MochiResponse:
    """Response dari Mochi Bot"""
    response_type: ResponseType
    message: str
    function_call: Optional[Dict[str, Any]] = None
    audio_file: Optional[str] = None
    latency_ms: float = 0.0


# ============================================================
# MOCHI BOT MAIN CLASS
# ============================================================

class MochiBot:
    """Main Mochi Bot Class"""
    
    def __init__(self, config: MochiConfig):
        self.config = config
        self.conversation_history = []
        self.llm: Optional[MochiLLM] = None
        self.tts: Optional[TTSService] = None
        self.stt: Optional[STTService] = None
        self.dispatcher: Optional[ActionDispatcher] = None
        self._initialized = False
    
    def initialize(self, load_stt: bool = False) -> "MochiBot":
        """Initialize semua komponen
        
        Args:
            load_stt: Apakah load STT model (butuh waktu & memory)
        """
        print("\n" + "=" * 60)
        print("🤖 MOCHI BOT INITIALIZATION")
        print("=" * 60)
        
        # 1. Initialize LLM
        print("\n📦 Loading LLM Model...")
        self.llm = MochiLLM(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers
        )
        self.llm.load()
        
        # 2. Initialize TTS
        print("\n🔊 Initializing TTS...")
        self.tts = TTSService(
            voice=self.config.tts_voice,
            rate=self.config.tts_rate,
            cache_dir=os.path.join(self.config.project_dir, "audio_cache")
        )
        
        # 3. Initialize STT (optional, only for audio mode)
        if load_stt:
            print("\n🎤 Loading STT Model...")
            self.stt = STTService(
                model_size=self.config.stt_model,
                device=self.config.stt_device,
                compute_type=self.config.stt_compute_type,
                language="id"
            )
            self.stt.load()
        
        # 4. Initialize Action Dispatcher
        print("\n📱 Initializing Action Dispatcher...")
        self.dispatcher = ActionDispatcher()
        
        self._initialized = True
        
        print("\n" + "=" * 60)
        print("✅ MOCHI BOT READY!")
        print("=" * 60)
        print(f"   👤 User: {self.config.user_name}")
        print(f"   📞 Panggilan: {self.config.preferred_name}")
        print(f"   👩‍⚕️ Caregiver: {self.config.caregiver_name}")
        print(f"   🎤 STT: {'Loaded' if load_stt else 'Not loaded'}")
        print("=" * 60 + "\n")
        
        return self
    
    def _build_system_prompt(self) -> str:
        """Build system prompt dengan context"""
        return SYSTEM_PROMPT_ID.format(
            user_name=self.config.user_name,
            preferred_name=self.config.preferred_name,
            caregiver_name=self.config.caregiver_name,
            current_time=datetime.now().strftime("%H:%M")
        )
    
    def _get_confirmation_message(self, function:  str, args: dict) -> str:
        """Generate pesan konfirmasi dalam Bahasa Indonesia"""
        caregiver = self.config.caregiver_name
        name = self.config.preferred_name
        
        if function == "alert_caregiver":
            return f"🚨 Saya SEGERA menghubungi {caregiver}! Tetap tenang ya {name}, bantuan akan segera datang."
        
        elif function == "request_service":
            details = args.get("details", "")
            service = args.get("service_type", "")
            
            if service == "drink": 
                item = details if details else "minuman"
                return f"☕ Baik {name}, saya sampaikan ke {caregiver} untuk menyiapkan {item}."
            elif service == "food":
                item = details if details else "makanan"
                return f"🍽️ Baik {name}, saya beritahu {caregiver} bahwa {name} ingin {item}."
            else:
                return f"📋 Baik {name}, saya sampaikan permintaan {name} ke {caregiver}."
        
        elif function == "request_assistance":
            assist_type = args.get("assistance_type", "")
            urgency = args.get("urgency", "normal")
            
            urgency_text = " dengan segera" if urgency == "urgent" else ""
            
            if assist_type == "toilet":
                return f"🚻 Baik {name}, saya sudah memberitahu {caregiver}{urgency_text}.  Beliau akan segera membantu."
            elif assist_type == "shower":
                return f"🚿 Baik {name}, {caregiver} akan segera datang membantu {name} mandi."
            else:
                return f"🤝 Saya sudah memberitahu {caregiver}. Beliau akan segera membantu {name}{urgency_text}."
        
        return f"✅ Baik {name}, saya akan sampaikan ke {caregiver}."
    
    async def process(self, user_text: str, generate_audio: bool = True) -> MochiResponse:
        """Process user input dan generate response"""
        if not self._initialized:
            raise RuntimeError("Bot belum diinisialisasi!  Panggil initialize() terlebih dahulu.")
        
        import time
        start_time = time. time()
        
        # Build messages
        messages = [
            {"role": "system", "content":  self._build_system_prompt()}
        ]
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": user_text})
        
        # Call LLM
        llm_response = self.llm.generate(
            messages=messages,
            tools=TOOLS_DEFINITION,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse response
        if llm_response.get("function_call"):
            func_name = llm_response["function_call"]["name"]
            func_args = llm_response["function_call"]["arguments"]
            
            confirmation = self._get_confirmation_message(func_name, func_args)
            
            result = MochiResponse(
                response_type=ResponseType.ACTION,
                message=confirmation,
                function_call={
                    "function": func_name,
                    "arguments": func_args
                }
            )
            
            # Dispatch action
            self.dispatcher.dispatch(result. function_call)
            
        else:
            content = llm_response.get("content", "")
            
            # Clean up jika content masih mengandung sisa function call syntax
            if content:
                content = self._clean_function_artifacts(content)
            
            if not content:
                content = f"Maaf {self.config.preferred_name}, saya kurang mengerti. Bisa diulang?"
            
            result = MochiResponse(
                response_type=ResponseType. CONVERSATION,
                message=content
            )
        
        # Update history
        self.conversation_history. append({"role": "user", "content":  user_text})
        self.conversation_history.append({"role": "assistant", "content": result.message})
        
        # Generate audio if requested
        if generate_audio:
            # Clean message for TTS (remove emoji)
            clean_message = self._clean_for_tts(result.message)
            result.audio_file = await self.tts.generate(clean_message)
        
        # Calculate latency
        result.latency_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _clean_for_tts(self, text: str) -> str:
        """Remove emoji and special chars for TTS"""
        import re
        # Remove emoji
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text).strip()
    
    def _clean_function_artifacts(self, text: str) -> str:
        """
        Bersihkan sisa-sisa function call syntax dari respons.
        Model kecil kadang mencampur respons dengan function syntax.
        """
        import re
        
        # Pattern untuk function call artifacts
        patterns_to_remove = [
            r'functions?\.(alert_caregiver|request_service|request_assistance)[:\s]*',
            r'<function[=\s]*(alert_caregiver|request_service|request_assistance)[^>]*>',
            r'</?tool_call>',
            r'</?function>',
            r'\{["\']?name["\']?\s*:\s*["\']?(alert_caregiver|request_service|request_assistance)["\']?[^}]*\}',
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Bersihkan whitespace berlebih
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️ Riwayat percakapan dihapus.")
    
    def process_sync(self, user_text: str, generate_audio: bool = True) -> MochiResponse:
        """Synchronous wrapper untuk process()"""
        return asyncio.run(self.process(user_text, generate_audio))
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file ke text menggunakan STT"""
        if self.stt is None:
            raise RuntimeError("STT belum di-load! Panggil initialize(load_stt=True)")
        
        result = self.stt.transcribe_file(audio_path)
        return result.text
    
    def run_server(self):
        """Run FastAPI server"""
        from api.server import run_server
        
        run_server(
            model_path=self.config.model_path,
            project_dir=self.config.project_dir,
            host=self.config.server_host,
            port=self.config.server_port,
            user_name=self.config.user_name,
            preferred_name=self.config.preferred_name,
            caregiver_name=self.config.caregiver_name
        )


# ============================================================
# INTERACTIVE CLI
# ============================================================

def run_interactive_cli():
    """Run interactive CLI mode"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🤖 MOCHI BOT - VERSI INDONESIA 🇮🇩               ║
    ║                                                              ║
    ║         Teman AI untuk Lansia - Local Deployment             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize
    config = MochiConfig()
    bot = MochiBot(config)
    bot.initialize()
    
    print("\n💬 MODE CHAT INTERAKTIF")
    print("   Ketik pesan dalam Bahasa Indonesia")
    print("   Ketik 'reset' untuk hapus riwayat")
    print("   Ketik 'keluar' untuk berhenti")
    print("-" * 50)
    
    while True:
        try: 
            user_input = input(f"\n👴 {config.preferred_name}:  ").strip()
            
            if not user_input:
                continue
            
            if user_input. lower() == 'reset':
                bot.clear_history()
                continue
            
            if user_input.lower() in ['keluar', 'exit', 'quit']:
                print("\n👋 Sampai jumpa!  Semoga sehat selalu.\n")
                break
            
            # Process
            result = bot.process_sync(user_input, generate_audio=False)
            
            print(f"🤖 Mochi:  {result.message}")
            print(f"   ⏱️ Latency: {result.latency_ms:.0f}ms")
            
            if result.function_call:
                print(f"   📱 Action: {result.function_call['function']}")
            
        except KeyboardInterrupt: 
            print("\n\n👋 Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mochi Bot - Elderly Care AI Companion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  cli     - Interactive text chat (default)
  server  - Run FastAPI server for ESP32/API access

Examples:
  python src/main.py                    # CLI mode
  python src/main.py --mode server      # Server mode
  python src/main.py --mode cli         # CLI mode explicit
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cli", "server"],
        default="cli",
        help="Run mode: cli (text chat) or server (API)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        run_interactive_cli()
    elif args.mode == "server":
        config = MochiConfig()
        config.server_host = args.host
        config.server_port = args.port
        
        bot = MochiBot(config)
        bot.run_server()