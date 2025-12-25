"""
STABLE VOICE CONVERSATIONAL AI (Wake-Word + Audio Cues)
Optimizations:
- VAD (Voice Activity Detection): Stop recording early when silence is detected.
- Silent Wake-Word Detection: Back-to-life on "Hey", "Fren", or "Continue".
- Audio Cues: Beep on pause/resume.
"""
import os
import sys

# Log suppression
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['ORT_CUDA_DISABLE_CUDNN_FRONTEND'] = '1'

import asyncio
import requests
import numpy as np
import soundfile as sf
import subprocess
import pyaudio
import tempfile
import threading
import re
import json
import logging
import warnings
import time
from datetime import datetime
from faster_whisper import WhisperModel

# Global Silence Configuration
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("lmdeploy").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Fix for CUDA DLLs on Windows
if sys.platform == "win32":
    for venv_name in [".venv", "venv"]:
        nvidia_dir = os.path.join(os.path.dirname(__file__), venv_name, "Lib", "site-packages", "nvidia")
        if os.path.exists(nvidia_dir):
            for sub in ["cudnn", "cublas", "cuda_runtime", "curand", "cusolver", "cusparse"]:
                bin_path = os.path.join(nvidia_dir, sub, "bin")
                if os.path.exists(bin_path):
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

# Set ONNX Severity
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except: pass

os.environ['TM_MAX_CONTEXT_TOKEN_NUM'] = '12000'
from mira.model import MiraTTS

# Configuration
LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODELS_URL = "http://127.0.0.1:1234/v1/models"
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 6 
SILENCE_THRESHOLD = 500  # RMS threshold for silence
SILENCE_DURATION = 0.8   # Stop after 0.8s of silence

class VoiceConversationalAI:
    def __init__(self, reference_audio_path):
        print("\n" + "="*70)
        print("🎙️  VOICE CONVERSATIONAL AI SYSTEM")
        print("="*70)
        
        # Initialize Whisper
        print("\n[1/4] Loading Whisper model...")
        try:
            self.whisper = WhisperModel("medium", device="cuda", compute_type="float16")
            print("      ✓ Whisper loaded on CUDA (medium)")
        except:
            self.whisper = WhisperModel("small", device="cpu", compute_type="int8")
            print("      ✓ Whisper loaded on CPU (small)")

        # Initialize TTS
        print("\n[2/4] Loading MiraTTS model...")
        self.mira_tts = MiraTTS('YatharthS/MiraTTS')
        self.mira_tts.set_params(temperature=0.6, top_p=0.92, max_new_tokens=1536)
        print("      ✓ MiraTTS loaded")
        
        print("\n[3/4] Encoding reference audio...")
        self.context_tokens = self.mira_tts.encode_audio(reference_audio_path)
        print("      ✓ Reference audio encoded")
        
        self.audio = pyaudio.PyAudio()
        self.turn_count = 0
        
        print("\n" + "="*70)
        print("[OK] System Ready!")
        print("="*70)

    def play_cue(self, cue_type='resume'):
        """Simple sine wave beep cue"""
        def run_cue():
            try:
                duration = 0.2
                fs = 44000
                if cue_type == 'resume':
                    # Ascending: 400Hz to 800Hz
                    f1, f2 = 400, 800
                else:
                    # Descending: 800Hz to 400Hz
                    f1, f2 = 800, 400
                
                t = np.linspace(0, duration, int(fs * duration))
                # Frequency sweep
                f = np.linspace(f1, f2, len(t))
                samples = 0.9 * np.sin(2 * np.pi * f * t) # Increased volume to 0.9
                
                # Fade out to avoid click
                fade = np.linspace(1, 0, len(samples))
                samples = (samples * fade).astype(np.float32)
                
                # Standard playback
                stream = self.audio.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
                stream.write(samples.tobytes())
                stream.stop_stream()
                stream.close()
            except: pass
        
        threading.Thread(target=run_cue).start()

    async def record_audio_vad(self, silent=False):
        """Record with VAD: stop early when silence is detected"""
        if not silent: print(f"\nMIC: Listening...")
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=3, 
            frames_per_buffer=1024
        )
        
        frames = []
        silent_chunks = 0
        max_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / 1024)
        started_talking = False
        
        for i in range(0, int(SAMPLE_RATE / 1024 * (3 if silent else MAX_RECORD_SECONDS))):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            
            if rms > SILENCE_THRESHOLD:
                started_talking = True
                silent_chunks = 0
            elif started_talking:
                silent_chunks += 1
                if silent_chunks > max_silent_chunks:
                    break
        
        stream.stop_stream()
        stream.close()
        return b''.join(frames)
    
    def transcribe_sync(self, audio_bytes, silent=False):
        """Standard transcription or silent background check"""
        if not silent: print("🔄 Transcribing...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_name = f.name
            with sf.SoundFile(temp_name, 'w', 16000, 1) as sf_file:
                sf_file.write(np.frombuffer(audio_bytes, dtype=np.int16))
        
        clean_name = temp_name.replace(".wav", "_clean.wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", temp_name, "-ac", "1", "-ar", "16000", "-af", "loudnorm,silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB", clean_name], capture_output=True, timeout=5)
            trans_file = clean_name if os.path.exists(clean_name) else temp_name
        except: trans_file = temp_name

        try:
            segments, _ = self.whisper.transcribe(trans_file, language="en", temperature=0, beam_size=1, vad_filter=True)
            text = "".join(s.text for s in segments).strip()
            
            if not silent:
                if text: print(f"   ✓ You said: \"{text}\"")
                else: print(f"   WARN: No speech detected")
            
            return text
        except: return ""
        finally:
            for f in [temp_name, clean_name]:
                if os.path.exists(f): 
                    try: os.remove(f)
                    except: pass

    async def background_wait_for_wake_word(self, initial=False):
        """Silently listen for 'Hey', 'Fren', or 'Continue'"""
        if initial:
            print("\n🎙️  System initialized. Say 'Hey Fren' or 'Start' to begin...")
        else:
            self.play_cue('pause') # Audio cue for pausing
            print("\n⏸️  System Paused. Listening for 'Hey', 'Fren', or 'Continue'...")
        
        keywords = ['hey', 'fren', 'continue', 'start']
        
        while True:
            # Silent record (3 seconds chunks)
            audio_data = await self.record_audio_vad(silent=True)
            
            # Silent transcribe
            text = await asyncio.to_thread(self.transcribe_sync, audio_data, silent=True)
            
            if text and any(kw in text.lower() for kw in keywords):
                self.play_cue('resume') # Audio cue for resuming
                print("\n🎙️  System Re-activated!" if not initial else "\n🎙️  System Started!")
                return True
            
            await asyncio.sleep(0.1)

    def query_llm_sync(self, user_message):
        try:
            try: model_id = requests.get(LLM_MODELS_URL, timeout=5).json()["data"][0]["id"]
            except: model_id = "local-model"

            response = requests.post(
                LLM_API_URL,
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a helpful voice assistant. Respond naturally and conversationally. Keep it concise (1-3 sentences). Only output plain spoken English."},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.5
                },
                timeout=60
            )
            if response.status_code == 200:
                msg = response.json()['choices'][0]['message']['content']
                msg = re.sub(r"<[^>]+>", "", msg)
                msg = re.sub(r"\*[^*]+\*", "", msg)
                return msg.strip()
            return "I'm having trouble connecting."
        except Exception as e:
            return f"Error: {e}"

    def generate_and_play_speech(self, text):
        if not text: return
        print(f"\nSPEAKER: Generating speech...")
        
        try:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                audio = self.mira_tts.generate(text, self.context_tokens)

            def play():
                audio_int16 = (np.array(audio) * 32767).astype(np.int16)
                stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
                stream.write(audio_int16.tobytes())
                stream.stop_stream()
                stream.close()

            def type_text():
                time.sleep(1.0) # Delay start to sync with audio buffering
                print(f"\n💬 Fren: ", end="", flush=True)
                for char in text:
                    print(char, end="", flush=True)
                    time.sleep(0.04) # Faster typeface animation
                print("\n")

            play_thread = threading.Thread(target=play)
            type_thread = threading.Thread(target=type_text)
            
            play_thread.start()
            type_thread.start()
            
            play_thread.join()
            type_thread.join()
            
        except Exception as e:
            print(f"   [ERR] TTS Error: {e}")

    async def run(self):
        print("\n" + "="*70)
        print("🎙️  VOICE CONVERSATION MODE")
        print("="*70)
        
        # Start with silent VAD instead of Enter
        await self.background_wait_for_wake_word(initial=True)
        
        try:
            while True:
                print("\n" + "-"*40)
                # Step 1: Record with VAD
                audio_data = await self.record_audio_vad()
                
                # Step 2: Transcribe
                transcription = await asyncio.to_thread(self.transcribe_sync, audio_data)
                
                if transcription:
                    # Check for termination
                    if any(word in transcription.lower() for word in ['deactivate', 'terminate', 'shut down']):
                        print("\n🛑 System Deactivating. Goodbye!")
                        break

                    # Check for pause/exit
                    if any(word in transcription.lower() for word in ['exit', 'quit', 'goodbye', 'bye', 'pause']):
                        # Enter Silent Wake Mode
                        await self.background_wait_for_wake_word()
                        continue
                        
                    # Step 3: Query LLM
                    response = await asyncio.to_thread(self.query_llm_sync, transcription)
                    
                    # Step 4: Playback
                    await asyncio.to_thread(self.generate_and_play_speech, response)
                
                print("\n⏸️  Ready for next turn...")
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt: print("\n👋 Interrupt received. Exiting.")
        finally: self.audio.terminate()

async def main():
    ref_file = "reference_file.wav"
    if not os.path.exists(ref_file): return
    ai = VoiceConversationalAI(ref_file)
    await ai.run()

if __name__ == "__main__":
    asyncio.run(main())
