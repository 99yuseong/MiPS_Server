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
from dataSet import indexNum, HopSize, AudioFile, headRot, soundSource, NumElePosition, HRIR_L, HRIR_R, FS, FrameSize, fileList, outBufferLArray, outBufferRArray, inBufferArray, soundOutLArray, soundOutRArray

app = FastAPI()

HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource[0], NumElePosition)

current_task = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global HRIR_L_INT, HRIR_R_INT, headRot
     
    await websocket.accept()
    
    i = 0
    maxI = int(len(AudioFile[0]) / HopSize)
    
    while i < maxI:   
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
        
        start = time.time()
        
        json_data = await multi_process_HRTFEffect(i)
        # json_data = await single_process_HRTFEffect(i)
        
        end = time.time()
        print(i, "total time: ", end-start)
        await websocket.send_json(json_data)

        i += 1
        
    await websocket.close()
        
async def update_HRIR():
    global HRIR_L_INT, HRIR_R_INT
    
    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, 
                                                           HRIR_R, 
                                                           headRot, 
                                                           soundSource[0], 
                                                           NumElePosition)

def schedule_update_HRIR():
    global current_task
    
    if current_task:
        current_task.cancel()
    
    current_task = asyncio.create_task(update_HRIR())
    
async def process_audio(instrumentIndex, i):
    global HRIR_L_INT, HRIR_R_INT, outBufferLArray, outBufferRArray, inBufferArray, soundOutLArray, soundOutRArray, AudioFile
    start = time.time()
    j = instrumentIndex
    outBufferL = outBufferLArray[j]
    outBufferR = outBufferRArray[j]
    inBuffer = inBufferArray[j]
    
    soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = await HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile[j][i*HopSize:(i+1)*HopSize], inBuffer, outBufferL, outBufferR)
    
    soundOutLArray[j] = soundOutL
    soundOutRArray[j] = soundOutR
    inBufferArray[j] = inBuffer
    outBufferLArray[j] = outBufferL
    outBufferRArray[j] = outBufferR
    end = time.time()
    print("process audio: ", end-start)
    

async def single_process_HRTFEffect(i):
    global HRIR_L_INT, HRIR_R_INT, outBufferLArray, outBufferRArray, inBufferArray, soundOutLArray, soundOutRArray, AudioFile
    
    # 0: electric guitar
    # 1: piano
    # 2: vocal
    # 3: other instrunments
    # 4: drum
    # 5: bass
    instrumentIndex = 4
    process_audio(instrumentIndex, i)
        
    left_list = np.array(soundOutLArray[instrumentIndex]).flatten().tolist()
    right_list = np.array(soundOutRArray[instrumentIndex]).flatten().tolist()

    json_data = {
        "index": i,
        "left": left_list,
        "right": right_list
    }
    
    return json_data

async def multi_process_HRTFEffect(i):
    global soundOutLArray, soundOutRArray, AudioFile

    tasks = []
    for index in range(len(AudioFile)):
        task = asyncio.create_task(process_audio(index, i))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    _left_list = [sum(sum(x)) for x in zip(*soundOutLArray)]
    _right_list = [sum(sum(x)) for x in zip(*soundOutRArray)]

    json_data = {
        "index": i,
        "left": _left_list,
        "right": _right_list
    }
    
    return json_data
    