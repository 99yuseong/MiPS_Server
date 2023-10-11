from fastapi import FastAPI, WebSocket
import asyncio
import pandas as pd
import librosa
import scipy
from scipy import signal

app = FastAPI()

FS = 44100
## Music data
AudioFile, FSAudio = librosa.load('soundhelix_mono.mp3')
SamplingNum = int(AudioFile.size * FS / FSAudio)
AudioFile = scipy.signal.resample(AudioFile, SamplingNum)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
        
    chunk_size = 1024
    
    for i in range(0, len(AudioFile), chunk_size):
        chunk1 = AudioFile[i:i+chunk_size].tobytes()
        chunk2 = AudioFile[i:i+chunk_size].tobytes()
        await websocket.send_bytes(chunk1 + chunk2)
    
    await websocket.close()