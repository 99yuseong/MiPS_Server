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
import json
import math

app = FastAPI()

FrameSize = 1024
FS = 44100
HopSize = int(FrameSize/2)

HRIR_L = pd.read_csv('Source/csv/HRIR_L.csv')
HRIR_R = pd.read_csv('Source/csv/HRIR_R.csv')

## Music data
AudioFile, FSAudio = librosa.load('Source/Music/Music_drum.mp3')
SamplingNum = int(AudioFile.size * FS / FSAudio)
AudioFile = scipy.signal.resample(AudioFile, SamplingNum)

remainder = len(AudioFile) % HopSize
if remainder != 0:
    pad_length = HopSize - remainder
    AudioFile = np.pad(AudioFile, (0, pad_length), 'constant', constant_values = 0)

## Buffer setup
outBufferL = np.zeros((FS + FrameSize, 1))
outBufferR = np.zeros((FS + FrameSize, 1))

inBuffer = np.zeros(FrameSize)
    
soundOutL = np.zeros((HopSize, 1))
soundOutR = np.zeros((HopSize, 1))

## Source Position
SourcePosition = pd.read_csv('Source/csv/SourcePosition.csv')
NumElePosition = HRTF_Effect.divideSourcePostion(SourcePosition)

headRot = [0, 0, 0]
soundSource = [0, 0]
# INTERVAL = HopSize / FS 
INTERVAL = 0.011

HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)

current_task = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global HRIR_L_INT, HRIR_R_INT, outBufferL, outBufferR, inBuffer, soundOutL, soundOutR, headRot
    
    await websocket.accept()
    
    for i in range(0, len(AudioFile), HopSize):        
        try:
            start = time.time()
            
            new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.004)
            
            if new_data and any(abs(headRot[j] - new_data[axis]) >= 15 for j,axis in enumerate(['roll', 'pitch', 'yaw'])):
                start1 = time.time()
                headRot = [new_data['roll'], new_data['pitch'], new_data['yaw']]
                schedule_update_HRIR()
                end1 = time.time()
                print("interpolation: ", end1 - start1)
            
        except asyncio.TimeoutError:
            pass
        
        soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile[i:i+HopSize], inBuffer, outBufferL, outBufferR)
        
        # left_bytes = struct.pack(f'{len(soundOutL)}f', *soundOutL.flatten())
        # right_bytes = struct.pack(f'{len(soundOutR)}f', *soundOutR.flatten())

        # await websocket.send_bytes(left_bytes + right_bytes)
        
        left_list = np.array(soundOutL).flatten().tolist()
        right_list = np.array(soundOutR).flatten().tolist()

        json_data = {
            "index": i / HopSize,
            "count": 1,
            "left": left_list,
            "right": right_list
        }
        # print(i / 512)
        await websocket.send_json(json_data)

        end = time.time()
        # await waitInterval(start, end)
        
    await websocket.close()
    
async def waitInterval(start, end):
    elapsed_time = end - start
        
    if elapsed_time < INTERVAL:
        await asyncio.sleep(INTERVAL - elapsed_time)
    else:
        print("time over", INTERVAL, elapsed_time)
        
async def update_HRIR():
    global HRIR_L_INT, HRIR_R_INT
    
    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)
    
def schedule_update_HRIR():
    global current_task
    
    if current_task:
        current_task.cancel()
    
    current_task = asyncio.create_task(update_HRIR())