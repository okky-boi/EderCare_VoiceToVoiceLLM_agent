"""
Speech-to-Text Service untuk Mochi Bot
Menggunakan faster-whisper dengan model base untuk Indonesian
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    """Hasil transkripsi audio"""
    text: str
    language: str
    confidence: float
    duration_ms: float


class STTService:
    """Speech-to-Text Service menggunakan faster-whisper"""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "id"
    ):
        """
        Initialize STT Service
        
        Args:
            model_size: Ukuran model (tiny, base, small, medium, large-v3)
            device: Device untuk inference (cuda atau cpu)
            compute_type: Tipe komputasi (float16, int8, float32)
            language: Bahasa default untuk transkripsi
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model: Optional[WhisperModel] = None
        
    def load(self) -> "STTService":
        """Load model Whisper ke memory"""
        print(f"🎤 Loading STT Service...")
        print(f"   📂 Model: {self.model_size}")
        print(f"   🎮 Device: {self.device}")
        print(f"   🔢 Compute: {self.compute_type}")
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        
        print(f"   ✅ STT Model loaded successfully!")
        return self
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transkripsi audio ke text
        
        Args:
            audio: Path ke file audio atau numpy array (float32, mono)
            sample_rate: Sample rate audio (default 16000 Hz)
            
        Returns:
            TranscriptionResult dengan text dan metadata
        """
        if self.model is None:
            raise RuntimeError("Model belum di-load! Panggil load() dulu.")
        
        start_time = time.time()
        
        # Jika numpy array, pastikan format benar
        if isinstance(audio, np.ndarray):
            # Pastikan float32 dan normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Normalize jika belum
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0
        
        # Transkripsi
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,  # Filter bagian tanpa suara
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )
        
        # Gabungkan semua segments
        text_parts = []
        total_confidence = 0.0
        segment_count = 0
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            # Avg log prob to confidence (approximate)
            total_confidence += np.exp(segment.avg_logprob)  # Fixed: avg_logprob
            segment_count += 1
        
        full_text = " ".join(text_parts).strip()
        avg_confidence = total_confidence / max(segment_count, 1)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            confidence=round(avg_confidence, 3),
            duration_ms=round(duration_ms, 2)
        )
    
    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Transkripsi dari file audio
        
        Args:
            file_path: Path ke file audio (wav, mp3, dll)
            
        Returns:
            TranscriptionResult
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
        
        return self.transcribe(str(path))
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2
    ) -> TranscriptionResult:
        """
        Transkripsi dari bytes audio (PCM)
        
        Args:
            audio_bytes: Raw PCM audio bytes
            sample_rate: Sample rate (default 16000)
            sample_width: Bytes per sample (2 = 16-bit)
            
        Returns:
            TranscriptionResult
        """
        # Convert bytes ke numpy array
        if sample_width == 2:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int8)
        
        # Convert ke float32 normalized
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        return self.transcribe(audio_float, sample_rate)
