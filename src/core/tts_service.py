"""
Text-to-Speech Service untuk Mochi Bot
Menggunakan Edge TTS (gratis, kualitas bagus)
"""

import os
import asyncio
import hashlib
import edge_tts
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
    
    def clear_cache(self):
        """Clear semua audio cache"""
        for file in self.cache_dir.glob("tts_*.mp3"):
            file.unlink()
        print(f"🗑️ Audio cache cleared:  {self.cache_dir}")