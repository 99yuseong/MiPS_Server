from fastapi import FastAPI, WebSocket
import asyncio
import pandas as pd
import librosa
import scipy
import HRTF_Effect
import time
import numpy as np
import sounddevice as sd
import struct

app = FastAPI()

FrameSize = 1024
FS = 44100
HopSize = int(FrameSize/2)

HRIR_L = pd.read_csv('Source/csv/HRIR_L.csv')
HRIR_R = pd.read_csv('Source/csv/HRIR_R.csv')

## Music data
AudioFile, FSAudio = librosa.load('Source/mp3/soundhelix_mono.mp3')
SamplingNum = int(AudioFile.size * FS / FSAudio)
AudioFile = scipy.signal.resample(AudioFile, SamplingNum)

# if AudioFile.size % HopSize != 0:
#     remain = HopSize - AudioFile.size % HopSize
#     AudioFile = np.append(AudioFile, np.zeros(remain), axis = 0)

## Source Position
SourcePosition = pd.read_csv('Source/csv/SourcePosition.csv')
NumElePosition = HRTF_Effect.divideSourcePostion(SourcePosition)

headRot = [0, 0, 0]
soundSource = [112, 40]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    ## Buffer setup
    outBufferNum = FS + FrameSize
    outBufferL = np.zeros((outBufferNum, 1))
    outBufferR = np.zeros((outBufferNum, 1))

    inBufferNum = FrameSize
    inBuffer = np.zeros(inBufferNum)
    
    soundOutL = np.zeros((HopSize, 1))
    soundOutR = np.zeros((HopSize, 1))
    
    first_data = await websocket.receive_json()
    headRot = [first_data['roll'], first_data['pitch'], first_data['yaw']]
    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)
    
    for i in range(0, len(AudioFile), HopSize):        
        try:
            start = time.time()
            
            new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.004)
            
            new_headRot = [new_data['roll'], new_data['pitch'], new_data['yaw']]
            
            if new_data and (
                abs(headRot[0] - new_headRot[0]) >= 1 
                or abs(headRot[1] - new_headRot[1]) >= 1 
                or abs(headRot[2] - new_headRot[2]) >= 1
            ):
                headRot = new_headRot
            
                HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)
            
            soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile[i:i+HopSize], inBuffer, outBufferL, outBufferR)

        except asyncio.TimeoutError:
            soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile[i:i+HopSize], inBuffer, outBufferL, outBufferR)
            pass
        
        left = b''.join(struct.pack('f', value) for value in [coord[0] for coord in soundOutL])
        right = b''.join(struct.pack('f', value) for value in [coord[0] for coord in soundOutR])

        await websocket.send_bytes(left + right)

        end = time.time()
        elapsed_time = end - start
    
        if elapsed_time < 0.01160997732:
            await asyncio.sleep(0.01160997732 - elapsed_time)
        print("time:", start - end)
        
    await websocket.close()