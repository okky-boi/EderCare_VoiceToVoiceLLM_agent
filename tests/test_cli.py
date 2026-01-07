"""
Mochi Bot - CLI Test Script
Test LLM + TTS tanpa STT (text input only)

Usage:
    python tests/test_cli.py
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Optional: sounddevice untuk play audio
try:
    import sounddevice as sd
    import numpy as np
    from scipy.io import wavfile
    import subprocess
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ sounddevice/scipy tidak tersedia, audio tidak akan diputar")
    print("   Install dengan: pip install sounddevice scipy")


def play_audio_file(file_path: str):
    """Play audio file menggunakan berbagai metode"""
    if not os.path.exists(file_path):
        print(f"❌ File tidak ditemukan: {file_path}")
        return
    
    # Method 1: Gunakan system default player (paling reliable di Windows)
    try:
        if sys.platform == "win32":
            os.startfile(file_path)
            return
    except Exception:
        pass
    
    # Method 2: ffplay (jika ada ffmpeg)
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path],
            check=True
        )
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Method 3: sounddevice (perlu convert dari mp3)
    if AUDIO_AVAILABLE and file_path.endswith('.wav'):
        try:
            sample_rate, data = wavfile.read(file_path)
            sd.play(data, sample_rate)
            sd.wait()
            return
        except Exception as e:
            print(f"⚠️ Gagal play dengan sounddevice: {e}")
    
    print(f"📁 Audio tersimpan di: {file_path}")


def run_cli_test():
    """Run interactive CLI test"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🧪 MOCHI BOT - CLI TEST MODE 🧪                      ║
    ║                                                              ║
    ║         Test LLM + TTS (tanpa Speech-to-Text)                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    from main import MochiBot, MochiConfig
    
    # Initialize bot (tanpa STT untuk test cepat)
    print("🔧 Initializing Mochi Bot...")
    config = MochiConfig()
    bot = MochiBot(config)
    bot.initialize(load_stt=False)  # Skip STT untuk test cepat
    
    print("\n💬 MODE TEST CLI")
    print("   Ketik pesan dalam Bahasa Indonesia")
    print("   Ketik 'audio on' untuk enable audio playback")
    print("   Ketik 'audio off' untuk disable audio playback")
    print("   Ketik 'reset' untuk hapus riwayat")
    print("   Ketik 'keluar' untuk berhenti")
    print("-" * 50)
    
    play_audio = False
    
    while True:
        try:
            user_input = input(f"\n👴 {config.preferred_name}: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == 'reset':
                bot.clear_history()
                continue
            
            if user_input.lower() == 'audio on':
                play_audio = True
                print("🔊 Audio playback: ON")
                continue
            
            if user_input.lower() == 'audio off':
                play_audio = False
                print("🔇 Audio playback: OFF")
                continue
            
            if user_input.lower() in ['keluar', 'exit', 'quit']:
                print("\n👋 Sampai jumpa! Semoga sehat selalu.\n")
                break
            
            # Process dengan/tanpa audio
            result = bot.process_sync(user_input, generate_audio=play_audio)
            
            print(f"🤖 Mochi: {result.message}")
            print(f"   ⏱️ Latency: {result.latency_ms:.0f}ms")
            
            if result.function_call:
                print(f"   📱 Action: {result.function_call['function']}")
                print(f"   📋 Args: {result.function_call['arguments']}")
            
            if result.audio_file and play_audio:
                print(f"   🔊 Playing audio...")
                play_audio_file(result.audio_file)
            
        except KeyboardInterrupt:
            print("\n\n👋 Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def test_scenarios():
    """Run predefined test scenarios"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🧪 MOCHI BOT - AUTOMATED TEST SCENARIOS 🧪           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    from main import MochiBot, MochiConfig
    
    config = MochiConfig()
    bot = MochiBot(config)
    bot.initialize(load_stt=False)
    
    # Test scenarios
    scenarios = [
        # Conversation biasa
        ("Halo Mochi, apa kabar?", "conversation", None),
        ("Cuaca hari ini bagaimana ya?", "conversation", None),
        
        # Emergency
        ("Tolong! Saya jatuh!", "action", "alert_caregiver"),
        ("Aduh sakit dada saya!", "action", "alert_caregiver"),
        
        # Service - Food/Drink
        ("Saya haus, minta teh dong", "action", "request_service"),
        ("Mochi, saya lapar", "action", "request_service"),
        
        # Assistance - Toilet/Shower
        ("Saya mau ke toilet", "action", "request_assistance"),
        ("Mochi, saya ingin mandi", "action", "request_assistance"),
    ]
    
    print("\n📋 Running test scenarios...\n")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for i, (text, expected_type, expected_function) in enumerate(scenarios, 1):
        bot.clear_history()  # Reset setiap test
        
        result = bot.process_sync(text, generate_audio=False)
        
        # Check result
        actual_type = result.response_type.value
        actual_function = result.function_call["function"] if result.function_call else None
        
        type_ok = actual_type == expected_type
        func_ok = actual_function == expected_function
        
        status = "✅ PASS" if (type_ok and func_ok) else "❌ FAIL"
        
        if type_ok and func_ok:
            passed += 1
        else:
            failed += 1
        
        print(f"\nTest {i}: {text}")
        print(f"  Expected: type={expected_type}, function={expected_function}")
        print(f"  Actual:   type={actual_type}, function={actual_function}")
        print(f"  Response: {result.message[:80]}...")
        print(f"  {status}")
    
    print("\n" + "-" * 70)
    print(f"\n📊 Results: {passed} passed, {failed} failed out of {len(scenarios)} tests")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mochi Bot CLI Test")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run automated test scenarios"
    )
    
    args = parser.parse_args()
    
    if args.auto:
        success = test_scenarios()
        sys.exit(0 if success else 1)
    else:
        run_cli_test()
