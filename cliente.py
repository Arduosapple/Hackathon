"""
Cliente WebSocket para grabar audio, enviar al servidor y reproducir respuestas

Librerías necesarias (instalar con pip):

- websockets      # Cliente WebSocket para comunicación con el servidor
- sounddevice    # Captura y reproducción de audio desde micrófono y altavoces
- soundfile      # Lectura/escritura de archivos de audio WAV
- numpy          # Procesamiento de datos numéricos
- asyncio        # Librería estándar de Python para programación asíncrona

Instalación recomendada (en consola):

    pip install websockets sounddevice soundfile numpy

Nota:

- sounddevice requiere que tu sistema tenga configurado correctamente los controladores de audio.
- En Windows, es posible que necesites instalar Microsoft Visual C++ Redistributable para que sounddevice funcione bien.
- asyncio viene con Python, no es necesario instalarlo.

"""

import asyncio
import websockets
import sounddevice as sd
import numpy as np
import soundfile as sf
import io

SERVER_URL = "ws://localhost:8000/call/12345"
SAMPLE_RATE = 16000
DURATION = 4  # segundos por turno

async def grabar_audio():
    print("🎙️ Grabando turno del usuario...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("📤 Turno grabado.")
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format='WAV')
    buffer.seek(0)
    return buffer.read()

def reproducir_audio(audio_bytes):
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    print("🔊 Reproduciendo respuesta...")
    sd.play(data, samplerate=sr)
    sd.wait()

async def cliente():
    async with websockets.connect(SERVER_URL, max_size=10_000_000) as websocket:
        print("📞 Conectado al servidor.")
        
        # Reproducir mensaje de bienvenida
        bienvenida = await websocket.recv()
        reproducir_audio(bienvenida)

        while True:
            audio_bytes = await grabar_audio()
            await websocket.send(audio_bytes)

            respuesta_binaria = await websocket.recv()
            reproducir_audio(respuesta_binaria)

if __name__ == "__main__":
    asyncio.run(cliente())


