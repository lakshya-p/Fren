# Complete Conversational AI Pipeline Setup

## Architecture Overview

```
Audio Input → Whisper (STT) → LM Studio (LLM) → MiraTTS (TTS) → Audio Output
```

## Components

### 1. Whisper WebSocket Server (Speech-to-Text)

- **File**: `ws_whisper_server.py`
- **Endpoint**: `ws://127.0.0.1:8000/ws/transcribe`
- **Status**: ✅ RUNNING
- **Model**: faster-whisper (small, CUDA)
- **Environment**: `venv`

### 2. LM Studio (Language Model)

- **Endpoint**: `http://127.0.0.1:1234/v1/chat/completions`
- **Status**: ✅ RUNNING
- **Available Models**:
  - orpheus-3b-0.1-ft
  - llama-3.1-8b-lexi-uncensored-v2

### 3. MiraTTS (Text-to-Speech)

- **File**: `tts_test.py`
- **Environment**: `.venv` (with dot)
- **GPU Enforcement**: ✅ STRICT (ONNX CUDA only)
- **Reference Audio**: `reference_file.wav`

### 4. Pipeline Integration

- **File**: `whisper_to_llm_pipeline.py`
- **Environment**: `venv`
- **Features**:
  - Interactive text chat with LLM
  - Audio file transcription via Whisper
  - Complete STT → LLM flow

## Running the Complete Pipeline

### Step 1: Start Whisper Server (Already Running)

```powershell
.\venv\Scripts\Activate.ps1
uvicorn ws_whisper_server:app --host 0.0.0.0 --port 8000
```

### Step 2: Ensure LM Studio is Running (Already Running)

- LM Studio should be running on port 1234
- Load your preferred model

### Step 3: Run the Pipeline

```powershell
.\venv\Scripts\Activate.ps1
python whisper_to_llm_pipeline.py
```

**Interactive Mode Commands:**

- Type any message → sends directly to LLM
- `audio <filepath>` → transcribes audio then sends to LLM
- `clear` → clears conversation history
- `exit` → quit

### Step 4: Add TTS Output (Use tts_test.py separately)

For now, use `tts_test.py` in the `.venv` environment to generate speech from LLM responses.

## Environment Details

### `venv` (Whisper + Pipeline)

Packages installed:

- faster-whisper
- fastapi
- uvicorn
- websockets
- soundfile
- numpy
- requests
- pyaudio

### `.venv` (MiraTTS)

Packages installed:

- mira (MiraTTS)
- onnxruntime-gpu (CUDA)
- soundfile
- numpy
- requests

## GPU Requirements

### tts_test.py - STRICT GPU ENFORCEMENT ✅

```python
# Force ONNX Runtime to use CUDA only - fail if CUDA is not available
os.environ['ORT_CUDA_UNAVAILABLE'] = '0'

# Verify ONNX Runtime has CUDA before proceeding
has_gpu = any(p in providers for p in ['CUDAExecutionProvider', 'TensorrtExecutionProvider'])
if not has_gpu:
    print("ERROR: No GPU provider available for ONNX Runtime!")
    sys.exit(1)
```

### ws_whisper_server.py - GPU Configured

```python
model = WhisperModel(
    "small",
    device="cuda",           # GPU only
    compute_type="float16"
)
```

## Testing the Pipeline

### Test 1: Text to LLM

```
> Hello, how are you?
🤖 Sending to LM Studio...
✓ Assistant: [LLM Response]
```

### Test 2: Audio to LLM

```
> audio reference_file.wav
🎤 Transcribing: reference_file.wav
   [Whisper]: [Transcribed text]
✓ Transcription complete: "[text]"
🤖 Sending to LM Studio...
✓ Assistant: [LLM Response]
```

### Test 3: Full Pipeline (Manual)

1. Record/prepare audio file
2. Run: `audio myaudio.wav` in pipeline
3. Get transcription → LLM response
4. Copy LLM response
5. Run `tts_test.py` in `.venv` environment
6. Paste response to generate speech

## Current Status

✅ Whisper Server: RUNNING on port 8000
✅ LM Studio: RUNNING on port 1234  
✅ Pipeline Script: READY
✅ TTS (tts_test.py): READY with GPU enforcement
⏳ Full integration: Requires merging environments or API bridge

## Next Steps for Full Integration

To create a single script with all three components:

1. Install MiraTTS in the `venv` environment, OR
2. Install Whisper packages in the `.venv` environment, OR
3. Create a REST API wrapper around tts_test.py for the pipeline to call
