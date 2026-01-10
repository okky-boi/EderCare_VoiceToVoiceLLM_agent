"""
Text-to-Speech Service untuk Mochi Bot
Menggunakan Edge TTS (gratis, kualitas bagus)
"""

import os
import asyncio
import hashlib
import io
import edge_tts
import numpy as np
import librosa
from pathlib import Path
from typing import Optional


class TTSService:
    """TTS Service menggunakan Edge TTS"""
    
    def __init__(
        self,
        voice: str = "id-ID-GadisNeural",
        rate: str = "-5%",
        pitch: str = "+0Hz",
        cache_dir: str = "./audio_cache"
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self. cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   🔊 Voice: {voice}")
        print(f"   📁 Cache:  {cache_dir}")
    
    def _get_cache_path(self, text: str) -> Path:
        """Generate cache path berdasarkan text hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return self.cache_dir / f"tts_{text_hash}.mp3"
    
    async def generate(self, text: str, use_cache: bool = True) -> str:
        """Generate audio dari text"""
        
        # Check cache
        cache_path = self._get_cache_path(text)
        if use_cache and cache_path.exists():
            return str(cache_path)
        
        # Generate baru
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        await communicate.save(str(cache_path))
        return str(cache_path)
    
    def generate_sync(self, text: str, use_cache: bool = True) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self.generate(text, use_cache))
    
    async def generate_bytes(self, text: str) -> bytes:
        """Generate raw PCM audio bytes (signed 16-bit, little endian, 16000Hz, mono)"""
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        # Stream MP3 ke memory buffer
        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])
        
        mp3_buffer.seek(0)
        
        # Load MP3 menggunakan librosa dan convert ke PCM s16le
        try:
            # Load audio dari memory buffer
            audio, sr = librosa.load(mp3_buffer, sr=16000, mono=True)
            
            # Normalize dan convert ke int16
            # Clip to [-1, 1] range
            audio = np.clip(audio, -1.0, 1.0)
            
            # Convert to int16 (PCM signed 16-bit)
            audio_int16 = np.int16(audio * 32767)
            
            # Convert to bytes (little endian by default)
            pcm_bytes = audio_int16.tobytes()
            
            return pcm_bytes
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            raise Exception(f"Failed to convert audio to PCM format: {str(e)}")
    
    def clear_cache(self):
        """Clear semua audio cache"""
        for file in self.cache_dir.glob("tts_*.mp3"):
            file.unlink()
        print(f"🗑️ Audio cache cleared:  {self.cache_dir}")