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
# INTERVAL = HopSize / FS 
# INTERVAL = 0.011

HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)

current_task = None

# def receiveData(websocket: WebSocket):
#     receiveData_process = Process(target=receiveData_task, args=(websocket, ))
#     receiveData_process.start()

# async def receiveData_task(websocket: WebSocket):
#     while True:
#         try:
#             new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
#             if not queue.empty():
#                 queue.get()  # 기존 값이 있으면 제거함
#             queue.put(new_data)
#         except asyncio.TimeoutError:
#             pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global HRIR_L_INT, HRIR_R_INT, outBufferL, outBufferR, inBuffer, soundOutL, soundOutR, headRot
     
    await websocket.accept()
    
    # dataQueue = Queue()
    # audioQueue = Queue()
    
    # p = Process(target=audio_processing_task, args=(dataQueue, audioQueue))
    # p.start()
    # cal = 0
    # while True:
    #     try:
    #         data = await asyncio.wait_for(websocket.receive_json(), timeout=0.005)
    #         if not dataQueue.empty():
    #             dataQueue.get()
    #         dataQueue.put(data)
            
    #     except asyncio.TimeoutError:
    #         pass
        
    #     if not audioQueue.empty():
    #         json_data = audioQueue.get()
    #             # print(json_data)
    #         await websocket.send_json(json_data)
    #         print("send ", cal)
    #         cal += 1
        
    # p.join()
    
    # receiveData(websocket)
    
    i = 0
    while i < len(AudioFile):    
        try:
            # if not queue.empty():
            #     print(queue)
            # if new_data is None:
                
            new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
            _curIdx = new_data['curIndex']
            _headRot = new_data['headRotation']
            
            if new_data and any(abs(headRot[j] - _headRot[axis]) >= 10 for j,axis in enumerate(['roll', 'pitch', 'yaw'])):
                headRot = [_headRot['roll'], _headRot['pitch'], _headRot['yaw']]
                print(headRot)
                schedule_update_HRIR()
                print('cal', i, 'to', _curIdx)
                i = _curIdx
                print("interpolation: ", i)
            
        except asyncio.TimeoutError:
            pass
        
        soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile[i*HopSize:(i+1)*HopSize], inBuffer, outBufferL, outBufferR)
        
        
        left_list = np.array(soundOutL).flatten().tolist()
        right_list = np.array(soundOutR).flatten().tolist()

        json_data = {
            "index": i,
            "count": 1,
            "left": left_list,
            "right": right_list
        }
        # print(json_data)
        print(i)
        await websocket.send_json(json_data)

        end = time.time()
        i += 1
        
    await websocket.close()
    
# def audio_processing_task(dataQueue: Queue, audioQueue: Queue):
#     global HRIR_L_INT, HRIR_R_INT, outBufferL, outBufferR, inBuffer, soundOutL, soundOutR, headRot
    
#     curIndex = 0
#     headRot = [0, 0, 0]
#     audioLen = len(AudioFile)
    
#     while curIndex < audioLen:
#         if not dataQueue.empty():
#             data = dataQueue.get()
#             curIndex = data['curIndex'] * HopSize
#             headRot_data = data['headRotation']
            
#             if data and any(abs(headRot[j] - headRot_data[axis]) >= 15 for j,axis in enumerate(['roll', 'pitch', 'yaw'])):
#                 headRot = [headRot_data['roll'], headRot_data['pitch'], headRot_data['yaw']]
#                 schedule_update_HRIR()
                
#         soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, AudioFile[curIndex:curIndex+HopSize], inBuffer, outBufferL, outBufferR)
        
#         left_list = np.array(soundOutL).flatten().tolist()
#         right_list = np.array(soundOutR).flatten().tolist()

#         json_data = {
#             "index": curIndex / HopSize,
#             "count": 1,
#             "left": left_list,
#             "right": right_list
#         }
#         print(curIndex)
#         curIndex += HopSize
#         audioQueue.put(json_data)
#             # print(data)
    
# async def waitInterval(start, end):
#     elapsed_time = end - start
        
#     if elapsed_time < INTERVAL:
#         await asyncio.sleep(INTERVAL - elapsed_time)
#     else:
#         print("time over", INTERVAL, elapsed_time)
        
async def update_HRIR():
    global HRIR_L_INT, HRIR_R_INT
    
    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource, NumElePosition)
    
def schedule_update_HRIR():
    global current_task
    
    if current_task:
        current_task.cancel()
    
    current_task = asyncio.create_task(update_HRIR())