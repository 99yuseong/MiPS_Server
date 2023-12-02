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
from multiprocessing import Queue, Process

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

HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)

current_task = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global HRIR_L_INT, HRIR_R_INT, outBufferL, outBufferR, inBuffer, soundOutL, soundOutR, headRot
     
    await websocket.accept()
    
    i = 0
    while i < len(AudioFile):    
        try:    
            new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
            _curIdx = new_data['curIndex']
            _headRot = new_data['headRotation']
            
            if new_data and any(abs(headRot[j] - _headRot[axis]) >= 10 for j,axis in enumerate(['roll', 'pitch', 'yaw'])):
                headRot = [_headRot['roll'], _headRot['pitch'], _headRot['yaw']]
                schedule_update_HRIR()
                i = _curIdx
            
        except asyncio.TimeoutError:
            pass
        
        json_data = get_HRTFEffect(i)
        
        await websocket.send_json(json_data)

        i += 1
        
    await websocket.close()
        
async def update_HRIR():
    global HRIR_L_INT, HRIR_R_INT
    
    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)
    
def schedule_update_HRIR():
    global current_task
    
    if current_task:
        current_task.cancel()
    
    current_task = asyncio.create_task(update_HRIR())
    
def get_HRTFEffect(i):
    global HRIR_L_INT, HRIR_R_INT, outBufferL, outBufferR, inBuffer, soundOutL, soundOutR
    
    soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile[i*HopSize:(i+1)*HopSize], inBuffer, outBufferL, outBufferR)
        
    left_list = np.array(soundOutL).flatten().tolist()
    right_list = np.array(soundOutR).flatten().tolist()

    json_data = {
        "index": i,
        "left": left_list,
        "right": right_list
    }
    
    return json_data