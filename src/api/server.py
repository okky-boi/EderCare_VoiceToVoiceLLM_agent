"""
Mochi Bot - FastAPI Server
WebSocket dan REST API untuk ESP32 dan testing
"""

import os
import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

# Import Mochi components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stt_service import STTService, TranscriptionResult
from core.tts_service import TTSService
from core.audio_utils import AudioBuffer, SpeechDetector, AudioConfig
from core.llm_processor import MochiLLM
from core.action_dispatcher import ActionDispatcher
from prompts.system_prompts import SYSTEM_PROMPT_ID, TOOLS_DEFINITION


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    """Request untuk chat endpoint"""
    text: str
    user_id: str = "user_001"
    generate_audio: bool = True


class ChatResponse(BaseModel):
    """Response dari chat endpoint"""
    message: str
    response_type: str  # "conversation" atau "action"
    function_call: Optional[dict] = None
    audio_url: Optional[str] = None
    latency_ms: float


class TranscribeResponse(BaseModel):
    """Response dari transcribe endpoint"""
    text: str
    language: str
    confidence: float
    duration_ms: float


class HealthResponse(BaseModel):
    """Response health check"""
    status: str
    timestamp: str
    components: dict


# ============================================================
# MOCHI API CLASS
# ============================================================

class MochiAPI:
    """
    Mochi Bot API Controller
    Mengelola semua komponen dan endpoints
    """
    
    def __init__(
        self,
        model_path: str,
        project_dir: str = ".",
        user_name: str = "Nenek Sari",
        preferred_name: str = "Nenek",
        caregiver_name: str = "Ibu Rina"
    ):
        self.model_path = model_path
        self.project_dir = Path(project_dir)
        self.user_name = user_name
        self.preferred_name = preferred_name
        self.caregiver_name = caregiver_name
        
        # Components (lazy loaded)
        self.llm: Optional[MochiLLM] = None
        self.stt: Optional[STTService] = None
        self.tts: Optional[TTSService] = None
        self.dispatcher: Optional[ActionDispatcher] = None
        
        # Conversation history per user
        self.conversations: dict = {}
        
        # Audio cache directory
        self.audio_cache_dir = self.project_dir / "audio_cache"
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> "MochiAPI":
        """Initialize semua komponen"""
        print("\n" + "=" * 60)
        print("🌐 MOCHI API SERVER INITIALIZATION")
        print("=" * 60)
        
        # 1. Initialize LLM
        print("\n📦 Loading LLM Model...")
        self.llm = MochiLLM(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=-1
        ).load()
        
        # 2. Initialize STT
        print("\n🎤 Loading STT Model...")
        self.stt = STTService(
            model_size="base",
            device="cpu",  # Ganti dari "cuda" ke "cpu"
            compute_type="int8",  # Ganti dari "float16" ke "int8" (lebih cepat di CPU)
            language="id"
        ).load()
        
        # 3. Initialize TTS
        print("\n🔊 Initializing TTS...")
        self.tts = TTSService(
            voice="id-ID-GadisNeural",
            rate="-5%",
            cache_dir=str(self.audio_cache_dir)
        )
        
        # 4. Initialize Dispatcher
        print("\n📱 Initializing Action Dispatcher...")
        self.dispatcher = ActionDispatcher()
        
        print("\n" + "=" * 60)
        print("✅ MOCHI API READY!")
        print("=" * 60 + "\n")
        
        return self
    
    def _build_system_prompt(self) -> str:
        """Build system prompt dengan context"""
        return SYSTEM_PROMPT_ID.format(
            user_name=self.user_name,
            preferred_name=self.preferred_name,
            caregiver_name=self.caregiver_name,
            current_time=datetime.now().strftime("%H:%M")
        )
    
    def _get_confirmation_message(self, function: str, args: dict) -> str:
        """Generate pesan konfirmasi dalam Bahasa Indonesia"""
        caregiver = self.caregiver_name
        name = self.preferred_name
        
        print(f"🔧 Function Call: {function}, Args: {args}")
        
        if function == "alert_caregiver":
            return f"Saya SEGERA menghubungi {caregiver}! Tetap tenang ya {name}, bantuan akan segera datang."
        
        elif function == "request_service":
            details = args.get("details", "")
            service = args.get("service_type", "")
            
            if service == "drink":
                item = details if details else "minuman"
                return f"Baik {name}, saya sampaikan ke {caregiver} untuk menyiapkan {item}."
            elif service == "food":
                item = details if details else "makanan"
                return f"Baik {name}, saya beritahu {caregiver} bahwa {name} ingin {item}."
            else:
                return f"Baik {name}, saya sampaikan permintaan {name} ke {caregiver}."
        
        elif function == "request_assistance":
            assist_type = args.get("assistance_type", "")
            urgency = args.get("urgency", "normal")
            
            urgency_text = " dengan segera" if urgency == "urgent" else ""
            
            if assist_type == "toilet":
                return f"Baik {name}, saya sudah memberitahu {caregiver}{urgency_text}. Beliau akan segera membantu."
            elif assist_type == "shower":
                return f"Baik {name}, {caregiver} akan segera datang membantu {name} mandi."
            else:
                return f"Saya sudah memberitahu {caregiver}. Beliau akan segera membantu {name}{urgency_text}."
        
        # Default fallback jika function tidak dikenal
        return f"Baik {name}, saya akan sampaikan ke {caregiver}."
    
    def _clean_for_tts(self, text: str) -> str:
        """Remove emoji for TTS"""
        import re
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
    
    async def process_text(self, text: str, user_id: str, generate_audio: bool = True) -> ChatResponse:
        """Process text input dan generate response"""
        start_time = time.time()
        
        # Get/create conversation history
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        history = self.conversations[user_id]
        
        # Build messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": text})
        
        # Call LLM - llm_processor sudah handle parsing function call (termasuk malformed)
        llm_response = self.llm.generate(
            messages=messages,
            tools=TOOLS_DEFINITION,
            temperature=0.7,
            max_tokens=256
        )
        
        print(f"🤖 Raw LLM Response: {llm_response}")
        
        # Parse response - llm_processor mengembalikan dict dengan keys: content, function_call
        response_type = "conversation"
        function_call = None
        message = ""
        
        # LLM processor sudah parse function_call dengan benar (termasuk fallback untuk malformed)
        if llm_response.get("function_call"):
            func_name = llm_response["function_call"]["name"]
            func_args = llm_response["function_call"]["arguments"]
            
            print(f"📞 Function Call Detected: {func_name}, Args: {func_args}")
            
            # Generate confirmation message berdasarkan function dan arguments yang sebenarnya
            message = self._get_confirmation_message(func_name, func_args)
            response_type = "action"
            function_call = {
                "function": func_name,
                "arguments": func_args
            }
            
            # Dispatch action
            self.dispatcher.dispatch(function_call)
        
        # Normal conversation response
        else:
            message = llm_response.get("content", "")
            
            # Bersihkan jika masih ada sisa function syntax
            if message:
                message = self._clean_function_artifacts(message)
        
        # Fallback jika message kosong
        if not message:
            message = f"Maaf {self.preferred_name}, saya kurang mengerti. Bisa diulang?"
        
        # Update history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": message})
        
        # Keep only last 10 turns
        if len(history) > 20:
            self.conversations[user_id] = history[-20:]
        
        # Generate audio if requested
        audio_url = None
        if generate_audio:
            clean_message = self._clean_for_tts(message)
            audio_path = await self.tts.generate(clean_message)
            audio_filename = Path(audio_path).name
            audio_url = f"/api/audio/{audio_filename}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ChatResponse(
            message=message,
            response_type=response_type,
            function_call=function_call,
            audio_url=audio_url,
            latency_ms=round(latency_ms, 2)
        )
    
    async def transcribe_audio(self, audio_bytes: bytes) -> TranscribeResponse:
        """Transcribe audio bytes ke text"""
        result = self.stt.transcribe_bytes(audio_bytes)
        
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration_ms=result.duration_ms
        )
    
    async def generate_audio_bytes(self, text: str) -> bytes:
        """Generate audio bytes dari text tanpa menyimpan ke disk"""
        clean_message = self._clean_for_tts(text)
        return await self.tts.generate_bytes(clean_message)
    
    async def generate_audio_file(self, text: str) -> str:
        """Generate audio dan save ke file, return URL"""
        clean_message = self._clean_for_tts(text)
        audio_path = await self.tts.generate(clean_message)
        audio_filename = Path(audio_path).name
        return f"/api/audio/{audio_filename}"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

# Global API instance
mochi_api: Optional[MochiAPI] = None


def create_app(
    model_path: str,
    project_dir: str = ".",
    user_name: str = "Nenek Sari",
    preferred_name: str = "Nenek",
    caregiver_name: str = "Ibu Rina"
) -> FastAPI:
    """Create and configure FastAPI application"""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup and shutdown events"""
        global mochi_api
        
        # Startup
        print("🚀 Starting Mochi API Server...")
        mochi_api = MochiAPI(
            model_path=model_path,
            project_dir=project_dir,
            user_name=user_name,
            preferred_name=preferred_name,
            caregiver_name=caregiver_name
        )
        mochi_api.initialize()
        
        yield
        
        # Shutdown
        print("👋 Shutting down Mochi API Server...")
    
    app = FastAPI(
        title="Mochi Bot API",
        description="Elderly Care AI Companion - Indonesian Language",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # --------------------------------------------------------
    # REST ENDPOINTS
    # --------------------------------------------------------
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy" if mochi_api else "initializing",
            timestamp=datetime.now().isoformat(),
            components={
                "llm": mochi_api.llm is not None if mochi_api else False,
                "stt": mochi_api.stt is not None if mochi_api else False,
                "tts": mochi_api.tts is not None if mochi_api else False
            }
        )
    
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Chat endpoint - process text input
        
        Contoh request:
        {
            "text": "Saya haus, minta teh dong",
            "user_id": "nenek_001",
            "generate_audio": true
        }
        """
        if not mochi_api:
            raise HTTPException(status_code=503, detail="Server belum siap")
        
        return await mochi_api.process_text(
            text=request.text,
            user_id=request.user_id,
            generate_audio=request.generate_audio
        )
    
    @app.post("/api/transcribe", response_model=TranscribeResponse)
    async def transcribe(file: UploadFile = File(...)):
        """
        Transcribe audio file to text
        
        Accepts: WAV, MP3, atau raw PCM (16kHz, 16-bit, mono)
        """
        if not mochi_api:
            raise HTTPException(status_code=503, detail="Server belum siap")
        
        audio_bytes = await file.read()
        return await mochi_api.transcribe_audio(audio_bytes)
    
    @app.get("/api/audio/{filename}")
    async def get_audio(filename: str):
        """Serve audio file dari cache"""
        if not mochi_api:
            raise HTTPException(status_code=503, detail="Server belum siap")
        
        audio_path = mochi_api.audio_cache_dir / filename
        
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio tidak ditemukan")
        
        return FileResponse(
            path=str(audio_path),
            media_type="audio/mpeg",
            filename=filename
        )
    
    @app.post("/api/process-audio", response_model=ChatResponse)
    async def process_audio(request: Request, user_id: str = "user_001"):
        """
        Full pipeline: Audio Stream -> STT -> LLM -> TTS
        
        Accepts: Binary audio stream (application/octet-stream)
        Returns: JSON response dengan audio URL
        """
        if not mochi_api:
            raise HTTPException(status_code=503, detail="Server belum siap")
        
        # 1. Read binary audio stream
        audio_bytes = await request.body()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio stream tidak ditemukan")
        
        # 2. Transcribe audio
        transcription = await mochi_api.transcribe_audio(audio_bytes)
        print(f"📝 Transcription: {transcription.text}")
        
        # 3. Process with LLM
        response = await mochi_api.process_text(
            text=transcription.text,
            user_id=user_id,
            generate_audio=False
        )
        
        # 4. Generate audio file dan get URL
        audio_url = await mochi_api.generate_audio_file(response.message)
        response.audio_url = audio_url
        
        return response
    
    # --------------------------------------------------------
    # WEBSOCKET ENDPOINT
    # --------------------------------------------------------
    
    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket untuk real-time audio streaming
        
        Protocol:
        1. Client sends: {"type": "start"} - mulai recording
        2. Client sends: binary audio chunks (PCM 16kHz 16-bit mono)
        3. Client sends: {"type": "stop"} - selesai recording
        4. Server responds: {"type": "transcription", "text": "..."}
        5. Server responds: {"type": "response", "message": "...", "audio_url": "..."}
        """
        await websocket.accept()
        print("🔌 WebSocket connected")
        
        audio_buffer = AudioBuffer()
        speech_detector = SpeechDetector()
        user_id = f"ws_{id(websocket)}"
        
        try:
            while True:
                data = await websocket.receive()
                
                # Handle text messages (control)
                if "text" in data:
                    message = json.loads(data["text"])
                    msg_type = message.get("type", "")
                    
                    if msg_type == "start":
                        # Reset buffer untuk recording baru
                        audio_buffer.clear()
                        speech_detector.reset()
                        await websocket.send_json({"type": "ready"})
                        print("🎙️ Recording started")
                    
                    elif msg_type == "stop":
                        # Process accumulated audio
                        print("⏹️ Recording stopped")
                        
                        if not audio_buffer.is_empty():
                            # Transcribe
                            audio_float = audio_buffer.get_audio_float()
                            transcription = mochi_api.stt.transcribe(audio_float)
                            
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription.text,
                                "confidence": transcription.confidence
                            })
                            
                            print(f"📝 Transcription: {transcription.text}")
                            
                            # Process with LLM
                            response = await mochi_api.process_text(
                                text=transcription.text,
                                user_id=user_id,
                                generate_audio=True
                            )
                            
                            await websocket.send_json({
                                "type": "response",
                                "message": response.message,
                                "response_type": response.response_type,
                                "function_call": response.function_call,
                                "audio_url": response.audio_url,
                                "latency_ms": response.latency_ms
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Tidak ada audio yang direkam"
                            })
                        
                        # Reset for next recording
                        audio_buffer.clear()
                        speech_detector.reset()
                    
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                
                # Handle binary messages (audio data)
                elif "bytes" in data:
                    audio_chunk = data["bytes"]
                    audio_buffer.add_chunk(audio_chunk)
                    
                    # Optional: Check VAD untuk auto-stop
                    # import numpy as np
                    # chunk_np = np.frombuffer(audio_chunk, dtype=np.int16)
                    # status = speech_detector.process_chunk(chunk_np)
                    # if status['speech_complete']:
                    #     # Auto-trigger stop
                    #     pass
        
        except WebSocketDisconnect:
            print("🔌 WebSocket disconnected")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            await websocket.close()
    
    return app


# ============================================================
# RUN SERVER
# ============================================================

def run_server(
    model_path: str,
    project_dir: str = ".",
    host: str = "0.0.0.0",
    port: int = 8000,
    user_name: str = "Nenek Sari",
    preferred_name: str = "Nenek",
    caregiver_name: str = "Ibu Rina"
):
    """Run the FastAPI server"""
    import uvicorn
    
    app = create_app(
        model_path=model_path,
        project_dir=project_dir,
        user_name=user_name,
        preferred_name=preferred_name,
        caregiver_name=caregiver_name
    )
    
    print(f"\n🌐 Starting Mochi Bot Server...")
    print(f"   📍 Host: {host}")
    print(f"   🔌 Port: {port}")
    print(f"   📖 Docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Default configuration untuk testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Mochi Bot API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--project-dir", type=str, default=".", help="Project directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    run_server(
        model_path=args.model,
        project_dir=args.project_dir,
        host=args.host,
        port=args.port
    )
