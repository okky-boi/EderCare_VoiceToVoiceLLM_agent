"""
Mochi Bot - Full Audio Test Script
Test full pipeline: Mic -> STT -> LLM -> TTS -> Speaker

Usage:
    python tests/test_audio.py --local     # Test tanpa server (langsung load model)
    python tests/test_audio.py --server    # Test dengan server running

Requirements:
    pip install sounddevice scipy
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Check sounddevice availability
try:
    import sounddevice as sd
    import numpy as np
    from scipy.io import wavfile
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("❌ sounddevice atau scipy tidak tersedia!")
    print("   Install dengan: pip install sounddevice scipy")
    sys.exit(1)


# ============================================================
# AUDIO RECORDING
# ============================================================

class AudioRecorder:
    """Simple audio recorder using sounddevice"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self._stream = None
    
    def _callback(self, indata, frames, time_info, status):
        """Callback untuk stream recording"""
        if status:
            print(f"⚠️ Recording status: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def start(self):
        """Start recording"""
        self.audio_data = []
        self.recording = True
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=self._callback
        )
        self._stream.start()
        print("🎙️ Recording... (tekan Enter untuk stop)")
    
    def stop(self) -> np.ndarray:
        """Stop recording dan return audio data"""
        self.recording = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if not self.audio_data:
            return np.array([], dtype=np.int16)
        
        # Concatenate all chunks
        audio = np.concatenate(self.audio_data, axis=0)
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        print(f"⏹️ Recording stopped. Duration: {len(audio) / self.sample_rate:.2f}s")
        return audio
    
    def get_duration(self) -> float:
        """Get current recording duration in seconds"""
        if not self.audio_data:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self.audio_data)
        return total_samples / self.sample_rate


def play_audio_file(file_path: str):
    """Play audio file"""
    if not os.path.exists(file_path):
        print(f"❌ File tidak ditemukan: {file_path}")
        return
    
    try:
        # Method 1: Windows default player (paling reliable)
        if sys.platform == "win32":
            os.startfile(file_path)
            return
    except Exception:
        pass
    
    # Method 2: Try with sounddevice for WAV files
    if file_path.endswith('.wav'):
        try:
            sample_rate, data = wavfile.read(file_path)
            sd.play(data, sample_rate)
            sd.wait()
            return
        except Exception as e:
            print(f"⚠️ Gagal play dengan sounddevice: {e}")
    
    print(f"📁 Audio tersimpan di: {file_path}")


def play_audio_array(audio: np.ndarray, sample_rate: int = 16000):
    """Play audio from numpy array"""
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Gagal play audio: {e}")


# ============================================================
# LOCAL TEST (tanpa server)
# ============================================================

def test_local():
    """Test full pipeline secara lokal (load semua model)"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       🎤 MOCHI BOT - FULL AUDIO TEST (LOCAL) 🔊              ║
    ║                                                              ║
    ║       Mic -> STT -> LLM -> TTS -> Speaker                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    from main import MochiBot, MochiConfig
    
    print("🔧 Initializing Mochi Bot (dengan STT)...")
    print("   ⏳ Ini akan memakan waktu beberapa menit...\n")
    
    config = MochiConfig()
    bot = MochiBot(config)
    bot.initialize(load_stt=True)  # Load STT juga
    
    recorder = AudioRecorder(sample_rate=16000)
    
    print("\n" + "=" * 60)
    print("🎙️ AUDIO TEST MODE")
    print("=" * 60)
    print("   Tekan Enter untuk mulai recording")
    print("   Bicara dalam Bahasa Indonesia")
    print("   Tekan Enter lagi untuk stop dan process")
    print("   Ketik 'keluar' untuk berhenti")
    print("-" * 60)
    
    while True:
        try:
            cmd = input("\n>>> Tekan Enter untuk mulai, atau ketik 'keluar': ").strip().lower()
            
            if cmd in ['keluar', 'exit', 'quit']:
                print("\n👋 Sampai jumpa!")
                break
            
            # Start recording
            recorder.start()
            
            # Wait for Enter to stop
            input()  # Block until Enter pressed
            
            # Stop and get audio
            audio = recorder.stop()
            
            if len(audio) < 1600:  # Kurang dari 0.1 detik
                print("⚠️ Audio terlalu pendek, coba lagi")
                continue
            
            print("\n🔄 Processing...")
            
            # 1. STT
            print("   🎤 Transcribing...")
            start_time = time.time()
            
            audio_float = audio.astype(np.float32) / 32768.0
            stt_result = bot.stt.transcribe(audio_float)
            
            stt_time = (time.time() - start_time) * 1000
            print(f"   📝 Text: \"{stt_result.text}\"")
            print(f"   ⏱️ STT: {stt_time:.0f}ms")
            
            if not stt_result.text.strip():
                print("⚠️ Tidak ada text terdeteksi, coba bicara lebih jelas")
                continue
            
            # 2. LLM + TTS
            print("   🤖 Generating response...")
            result = bot.process_sync(stt_result.text, generate_audio=True)
            
            print(f"\n🤖 Mochi: {result.message}")
            print(f"   ⏱️ Total Latency: {result.latency_ms:.0f}ms")
            
            if result.function_call:
                print(f"   📱 Action: {result.function_call['function']}")
            
            # 3. Play audio response
            if result.audio_file:
                print(f"   🔊 Playing response...")
                play_audio_file(result.audio_file)
            
        except KeyboardInterrupt:
            print("\n\n👋 Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# SERVER TEST (dengan FastAPI server)
# ============================================================

def test_with_server(server_url: str = "http://localhost:8000"):
    """Test dengan server yang sudah running"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       🎤 MOCHI BOT - AUDIO TEST (SERVER MODE) 🔊             ║
    ║                                                              ║
    ║       Pastikan server sudah running:                         ║
    ║       python src/main.py --mode server                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    import httpx
    import io
    import wave
    
    # Check server availability
    print(f"🔌 Checking server at {server_url}...")
    try:
        with httpx.Client() as client:
            response = client.get(f"{server_url}/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ Server is running!")
            else:
                print(f"⚠️ Server returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Pastikan server sudah running dengan:")
        print(f"   python src/main.py --mode server")
        return
    
    recorder = AudioRecorder(sample_rate=16000)
    
    print("\n" + "=" * 60)
    print("🎙️ AUDIO TEST MODE (Server)")
    print("=" * 60)
    print("   Tekan Enter untuk mulai recording")
    print("   Bicara dalam Bahasa Indonesia")
    print("   Tekan Enter lagi untuk stop dan process")
    print("   Ketik 'keluar' untuk berhenti")
    print("-" * 60)
    
    while True:
        try:
            cmd = input("\n>>> Tekan Enter untuk mulai, atau ketik 'keluar': ").strip().lower()
            
            if cmd in ['keluar', 'exit', 'quit']:
                print("\n👋 Sampai jumpa!")
                break
            
            # Start recording
            recorder.start()
            
            # Wait for Enter to stop
            input()
            
            # Stop and get audio
            audio = recorder.stop()
            
            if len(audio) < 1600:
                print("⚠️ Audio terlalu pendek, coba lagi")
                continue
            
            print("\n🔄 Sending to server...")
            
            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio.tobytes())
            wav_buffer.seek(0)
            
            # Send to server
            start_time = time.time()
            
            with httpx.Client() as client:
                response = client.post(
                    f"{server_url}/api/process-audio",
                    files={"file": ("audio.wav", wav_buffer, "audio/wav")},
                    timeout=60.0
                )
            
            if response.status_code != 200:
                print(f"❌ Server error: {response.status_code}")
                print(f"   {response.text}")
                continue
            
            result = response.json()
            total_time = (time.time() - start_time) * 1000
            
            print(f"\n🤖 Mochi: {result['message']}")
            print(f"   ⏱️ Server Latency: {result['latency_ms']:.0f}ms")
            print(f"   ⏱️ Total (incl. network): {total_time:.0f}ms")
            
            if result.get('function_call'):
                print(f"   📱 Action: {result['function_call']['function']}")
            
            # Download and play audio
            if result.get('audio_url'):
                print(f"   🔊 Downloading audio...")
                audio_response = client.get(f"{server_url}{result['audio_url']}")
                if audio_response.status_code == 200:
                    # Save temporarily and play
                    temp_path = Path(__file__).parent / "temp_response.mp3"
                    temp_path.write_bytes(audio_response.content)
                    play_audio_file(str(temp_path))
            
        except KeyboardInterrupt:
            print("\n\n👋 Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# SIMPLE MIC TEST
# ============================================================

def test_microphone():
    """Test apakah microphone berfungsi"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🎤 MICROPHONE TEST 🎤                            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("📋 Available audio devices:")
    print(sd.query_devices())
    print()
    
    print("🎤 Recording 3 seconds...")
    duration = 3  # seconds
    sample_rate = 16000
    
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        
        # Calculate volume
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        
        print(f"✅ Recording complete!")
        print(f"   📊 RMS Volume: {rms:.4f}")
        print(f"   📏 Samples: {len(audio)}")
        
        if rms < 10:
            print("   ⚠️ Volume sangat rendah - cek microphone!")
        elif rms < 100:
            print("   📢 Volume rendah - bicara lebih keras")
        else:
            print("   ✅ Volume OK!")
        
        # Playback
        print("\n🔊 Playing back recording...")
        sd.play(audio, sample_rate)
        sd.wait()
        print("✅ Playback complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTips:")
        print("1. Pastikan microphone terhubung")
        print("2. Cek microphone permissions di Windows Settings")
        print("3. Coba restart terminal/VS Code")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mochi Bot - Full Audio Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/test_audio.py --local      # Test tanpa server
    python tests/test_audio.py --server     # Test dengan server
    python tests/test_audio.py --mic        # Test microphone saja
        """
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Test lokal (load semua model langsung)"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Test dengan server yang sudah running"
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Test microphone saja"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    if args.mic:
        test_microphone()
    elif args.server:
        test_with_server(args.url)
    elif args.local:
        test_local()
    else:
        # Default: local test
        print("💡 Tip: Gunakan --local, --server, atau --mic")
        print("   Menjalankan --local sebagai default...\n")
        test_local()