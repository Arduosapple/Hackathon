"""
Servidor FastAPI para sistema de atención telefónica con IA

Librerías necesarias (instalar con pip):

- fastapi                 # Framework web para API
- uvicorn[standard]       # Servidor ASGI para correr FastAPI
- openai-whisper          # Modelo Whisper para reconocimiento de voz (STT)
- TTS                     # Coqui TTS para síntesis de voz (TTS)
- transformers            # Modelos de lenguaje como LLaMA
- torch                   # PyTorch, backend para transformers
- soundfile               # Lectura/escritura de archivos de audio WAV
- librosa                 # Procesamiento y resampleo de audio
- numpy                   # Cálculos numéricos

Instalación recomendada (en consola):

    pip install fastapi "uvicorn[standard]" openai-whisper TTS transformers soundfile librosa numpy

Para instalar PyTorch, visitar:

    https://pytorch.org/get-started/locally/

y elegir la versión adecuada para tu sistema (CPU/GPU).
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import whisper
from TTS.api import TTS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import io
import soundfile as sf
import librosa
import asyncio
import logging
import difflib

app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call_center")

# Modelos
stt_model = whisper.load_model("small")
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# ⚠️ Carga de modelo LLaMA
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16
)
llm_model.eval()

calls_context = {}

# 📌 Respuestas predefinidas
PREDEFINED_RESPONSES = {
    "what time does the company close": "The company closes at 5:00 p.m.",
    "what time do you open": "We open at 8:00 a.m. from Monday to Friday.",
    "where are you located": "We are located at 123 Main Street, Springfield.",
    "can i speak to a human": "Sure, let me transfer you to a human agent."
}

def find_predefined_response(normalized_text):
    for key, value in PREDEFINED_RESPONSES.items():
        if key in normalized_text:
            return value
        similarity = difflib.SequenceMatcher(None, key, normalized_text).ratio()
        if similarity > 0.8:
            logger.info(f"Matching '{normalized_text}' to predefined key '{key}' with similarity {similarity:.2f}")
            return value
    return None

def generate_response(prompt, context):
    system_prompt = (
        "You are a helpful customer service assistant. "
        "Answer clearly and politely."
    )
    dialogue = context + f"\nUser: {prompt}\nAssistant:"
    full_prompt = system_prompt + "\n" + dialogue

    inputs = tokenizer(full_prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded_text:
        response = decoded_text.split("Assistant:")[-1].strip()
    else:
        response = decoded_text[len(full_prompt):].strip()  # fallback
    logger.info(f"Generated response: {response}")
    return response

def process_audio_to_text(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    audio, sample_rate = sf.read(audio_file)

    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    audio = audio.astype(np.float32)
    result = stt_model.transcribe(audio, fp16=False)
    return result["text"]

def text_to_speech(text):
    wav = tts_model.tts(text)
    if wav is None or len(wav) == 0:
        logger.error("⚠️ Audio vacío generado por TTS")
        return b""

    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=tts_model.synthesizer.output_sample_rate, format="WAV")
    buffer.seek(0)
    logger.info(f"✅ Audio generado: {len(wav)} samples")
    return buffer.read()

async def keep_alive(websocket: WebSocket, interval: int = 20):
    try:
        while True:
            await websocket.send_ping()
            await asyncio.sleep(interval)
    except Exception:
        pass

@app.websocket("/call/{call_id}")
async def call_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"Llamada {call_id} iniciada")
    calls_context[call_id] = {"history": [], "welcomed": False}

    welcome_text = "Thank you for calling our customer service. How can I help you today?"
    welcome_audio = text_to_speech(welcome_text)
    await websocket.send_bytes(welcome_audio)
    logger.info(f"[{call_id}] Mensaje de bienvenida enviado")
    calls_context[call_id]["welcomed"] = True

    keepalive_task = asyncio.create_task(keep_alive(websocket))

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            try:
                text = process_audio_to_text(audio_chunk)
                logger.info(f"[{call_id}] Texto reconocido: {text}")

                normalized = text.lower().strip()
                context = "\n".join(calls_context[call_id]["history"][-2:])

                if normalized in context.lower():
                    response_text = "I'm sorry, we have already discussed that. Would you like help with something else?"
                else:
                    response_text = find_predefined_response(normalized)

                    if not response_text:
                        response_text = generate_response(text, context)

                    calls_context[call_id]["history"].append(f"User: {text}")
                    calls_context[call_id]["history"].append(f"Agent: {response_text}")

                audio_response = text_to_speech(response_text)
                logger.info(f"[{call_id}] Enviando respuesta de audio: {len(audio_response)} bytes")
                await websocket.send_bytes(audio_response)
                logger.info(f"[{call_id}] Respuesta enviada")

            except Exception as e:
                logger.error(f"[{call_id}] Error interno: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"Llamada {call_id} desconectada")
    except Exception as e:
        logger.error(f"Llamada {call_id} error inesperado: {e}")
    finally:
        calls_context.pop(call_id, None)
        keepalive_task.cancel()
        logger.info(f"Llamada {call_id} finalizada")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=10485760)

