from fastapi import FastAPI, WebSocket
import asyncio
import pandas as pd
import librosa
import scipy
import HRTF_Effect
import time

app = FastAPI()

FrameSize = 1024
FS = 44100

HRIR_L = pd.read_csv('Source/csv/HRIR_L.csv')
HRIR_R = pd.read_csv('Source/csv/HRIR_R.csv')

## Music data
AudioFile, FSAudio = librosa.load('Source/mp3/soundhelix_mono.mp3')
SamplingNum = int(AudioFile.size * FS / FSAudio)
AudioFile = scipy.signal.resample(AudioFile, SamplingNum)

## Source Position
SourcePosition = pd.read_csv('Source/csv/SourcePosition.csv')
NumElePosition = HRTF_Effect.divideSourcePostion(SourcePosition)

headRot = [0, 0, 0]
soundSource = [112, 40]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    first_data = await websocket.receive_json()
    headRot = [first_data['roll'], first_data['pitch'], first_data['yaw']]
    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)
    
    for i in range(0, len(AudioFile), FrameSize):        
        try:
            new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.02)
            new_headRot = [new_data['roll'], new_data['pitch'], new_data['yaw']]
            
            if new_data and headRot != new_headRot:
                headRot = new_headRot
                HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)
                
            start = time.time()
            
            HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile)
            
            end = time.time()
            print("time", end-start)
            
        except asyncio.TimeoutError:
            pass
        
        # 여기서 HRTFEffect 함수
        left = AudioFile[i:i+FrameSize].tobytes()
        right = AudioFile[i:i+FrameSize].tobytes()
        
        await websocket.send_bytes(left + right)
        
    await websocket.close()