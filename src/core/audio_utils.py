"""
Audio Utilities untuk Mochi Bot
Energy-based VAD dan Audio Buffer untuk streaming
"""

import io
import wave
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AudioConfig:
    """Konfigurasi audio standard"""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit
    chunk_duration_ms: int = 100  # Durasi per chunk


class AudioBuffer:
    """
    Buffer untuk akumulasi audio chunks
    Digunakan untuk streaming audio dari ESP32/microphone
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        max_duration_sec: float = 30.0
    ):
        """
        Initialize audio buffer
        
        Args:
            config: Konfigurasi audio
            max_duration_sec: Maksimal durasi buffer (untuk prevent memory leak)
        """
        self.config = config or AudioConfig()
        self.max_samples = int(max_duration_sec * self.config.sample_rate)
        self._buffer: List[np.ndarray] = []
        self._total_samples = 0
        
    def add_chunk(self, chunk: bytes) -> None:
        """
        Tambah chunk audio ke buffer
        
        Args:
            chunk: Raw PCM bytes (16-bit signed)
        """
        # Convert bytes ke numpy
        audio_np = np.frombuffer(chunk, dtype=np.int16)
        
        # Cek max duration
        if self._total_samples + len(audio_np) > self.max_samples:
            # Trim dari awal jika melebihi max
            self._trim_buffer(len(audio_np))
        
        self._buffer.append(audio_np)
        self._total_samples += len(audio_np)
    
    def _trim_buffer(self, new_samples: int) -> None:
        """Trim buffer dari awal untuk make room"""
        samples_to_remove = self._total_samples + new_samples - self.max_samples
        
        while samples_to_remove > 0 and self._buffer:
            first_chunk = self._buffer[0]
            if len(first_chunk) <= samples_to_remove:
                self._buffer.pop(0)
                samples_to_remove -= len(first_chunk)
                self._total_samples -= len(first_chunk)
            else:
                # Partial trim
                self._buffer[0] = first_chunk[samples_to_remove:]
                self._total_samples -= samples_to_remove
                break
    
    def get_audio(self) -> np.ndarray:
        """
        Ambil semua audio sebagai numpy array (int16)
        
        Returns:
            numpy array int16
        """
        if not self._buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._buffer)
    
    def get_audio_float(self) -> np.ndarray:
        """
        Ambil audio sebagai float32 normalized (-1.0 to 1.0)
        Format yang dibutuhkan Whisper
        
        Returns:
            numpy array float32
        """
        audio = self.get_audio()
        return audio.astype(np.float32) / 32768.0
    
    def get_duration_ms(self) -> float:
        """Durasi audio dalam milliseconds"""
        return (self._total_samples / self.config.sample_rate) * 1000
    
    def clear(self) -> None:
        """Reset buffer"""
        self._buffer.clear()
        self._total_samples = 0
    
    def is_empty(self) -> bool:
        """Cek apakah buffer kosong"""
        return self._total_samples == 0
    
    @property
    def sample_count(self) -> int:
        """Jumlah sample dalam buffer"""
        return self._total_samples


def calculate_energy(audio: np.ndarray) -> float:
    """
    Hitung energy (RMS) dari audio
    
    Args:
        audio: numpy array audio (int16 atau float32)
        
    Returns:
        RMS energy value
    """
    if len(audio) == 0:
        return 0.0
    
    # Convert ke float jika int16
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    return float(np.sqrt(np.mean(audio ** 2)))


def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_silence_ms: int = 500,
    sample_rate: int = 16000
) -> bool:
    """
    Deteksi apakah akhir audio adalah silence
    Energy-based Voice Activity Detection (VAD)
    
    Args:
        audio: numpy array audio
        threshold: Energy threshold untuk silence (0.01 = -40dB)
        min_silence_ms: Minimum durasi silence untuk dianggap "selesai bicara"
        sample_rate: Sample rate audio
        
    Returns:
        True jika terdeteksi silence di akhir audio
    """
    if len(audio) == 0:
        return True
    
    # Hitung samples untuk min_silence
    min_silence_samples = int(min_silence_ms * sample_rate / 1000)
    
    if len(audio) < min_silence_samples:
        return False
    
    # Ambil bagian akhir audio
    tail = audio[-min_silence_samples:]
    energy = calculate_energy(tail)
    
    return energy < threshold


class SpeechDetector:
    """
    Detektor untuk menentukan kapan user selesai bicara
    Menggunakan energy-based VAD sederhana
    """
    
    def __init__(
        self,
        silence_threshold: float = 0.01,
        min_silence_ms: int = 500,
        min_speech_ms: int = 300,
        sample_rate: int = 16000
    ):
        """
        Initialize speech detector
        
        Args:
            silence_threshold: Threshold energy untuk silence
            min_silence_ms: Minimal durasi silence untuk dianggap selesai
            min_speech_ms: Minimal durasi speech sebelum bisa dianggap selesai
            sample_rate: Sample rate audio
        """
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.sample_rate = sample_rate
        
        # State tracking
        self._speech_started = False
        self._speech_start_sample = 0
        self._last_speech_sample = 0
        self._current_sample = 0
    
    def process_chunk(self, chunk: np.ndarray) -> dict:
        """
        Proses chunk audio dan update state
        
        Args:
            chunk: numpy array audio chunk
            
        Returns:
            dict dengan status: {
                'has_speech': bool,
                'speech_complete': bool,
                'energy': float
            }
        """
        energy = calculate_energy(chunk)
        chunk_has_speech = energy > self.silence_threshold
        
        if chunk_has_speech:
            if not self._speech_started:
                self._speech_started = True
                self._speech_start_sample = self._current_sample
            self._last_speech_sample = self._current_sample
        
        self._current_sample += len(chunk)
        
        # Check if speech is complete
        speech_complete = False
        if self._speech_started:
            # Cek apakah sudah cukup lama bicara
            speech_duration_samples = self._last_speech_sample - self._speech_start_sample
            min_speech_samples = int(self.min_speech_ms * self.sample_rate / 1000)
            
            if speech_duration_samples >= min_speech_samples:
                # Cek apakah sudah silence cukup lama
                silence_duration_samples = self._current_sample - self._last_speech_sample
                min_silence_samples = int(self.min_silence_ms * self.sample_rate / 1000)
                
                if silence_duration_samples >= min_silence_samples:
                    speech_complete = True
        
        return {
            'has_speech': chunk_has_speech,
            'speech_complete': speech_complete,
            'energy': energy
        }
    
    def reset(self) -> None:
        """Reset detector state"""
        self._speech_started = False
        self._speech_start_sample = 0
        self._last_speech_sample = 0
        self._current_sample = 0
    
    @property
    def is_speech_active(self) -> bool:
        """Apakah sedang dalam speech"""
        return self._speech_started


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 16000,
    sample_width: int = 2
) -> bytes:
    """
    Convert numpy array ke WAV bytes
    
    Args:
        audio: numpy array (int16)
        sample_rate: Sample rate
        sample_width: Bytes per sample
        
    Returns:
        WAV file as bytes
    """
    if audio.dtype != np.int16:
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    buffer.seek(0)
    return buffer.read()


def wav_bytes_to_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert WAV bytes ke numpy array
    
    Args:
        wav_bytes: WAV file as bytes
        
    Returns:
        Tuple of (numpy array int16, sample_rate)
    """
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
    
    return audio, sample_rate
