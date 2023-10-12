from fastapi import FastAPI, WebSocket
import asyncio
import pandas as pd
import librosa
import scipy
from scipy import signal

app = FastAPI()

FS = 44100
## Music data
AudioFile, FSAudio = librosa.load('Source/mp3/soundhelix_mono.mp3')
SamplingNum = int(AudioFile.size * FS / FSAudio)
AudioFile = scipy.signal.resample(AudioFile, SamplingNum)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    chunk_size = 1024
    
    for i in range(0, len(AudioFile), chunk_size):
        
        # 여기서 HRTFEffect 함수
        left = AudioFile[i:i+chunk_size].tobytes()
        right = AudioFile[i:i+chunk_size].tobytes()
        
        await websocket.send_bytes(left + right)
        
        try:
            new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.02)

            if new_data:
                print(new_data)
                # 후처리
            
        except asyncio.TimeoutError:
            pass
        
    await websocket.close()