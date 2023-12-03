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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from dataSet import indexNum, HopSize, AudioFile, headRot, soundSource, NumElePosition, HRIR_L, HRIR_R, FS, FrameSize, fileList, outBufferLArray, outBufferRArray, inBufferArray, soundOutLArray, soundOutRArray

app = FastAPI()

HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource[0], NumElePosition)

current_task = None
executor_thread = ThreadPoolExecutor(max_workers=len(AudioFile))
executor_process = ProcessPoolExecutor(max_workers=len(AudioFile))
loop = asyncio.get_event_loop()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global HRIR_L_INT, HRIR_R_INT, headRot, executor_thread, executor_process, loop
     
    await websocket.accept()
    
    i = 0
    maxI = int(len(AudioFile[0]) / HopSize)
    
    while i < maxI:   
        start = time.time()
        try:    
            new_data = await asyncio.wait_for(websocket.receive_json(), timeout=0.00005)
            
            headRot = [new_data['headRotation']['roll'], new_data['headRotation']['pitch'], new_data['headRotation']['yaw']]
            schedule_update_HRIR()
            i = new_data['curIndex']
            
        except asyncio.TimeoutError:
            pass
        
        # json_data = await single_HRTFEffect(i)
        json_data = await async_HRTFEffect(i)
        # json_data = await asyncio.to_thread(multi_thread_HRTFEffect, i)
        # json_data = await loop.run_in_executor(executor_process, multi_process_HRTFEffect, i)
        
        end = time.time()
        print(i, "total time: ", end-start)
        await websocket.send_json(json_data)

        i += 1
        
    await websocket.close()
        
async def update_HRIR():
    global HRIR_L_INT, HRIR_R_INT, headRot, soundSource, NumElePosition
    
    start = time.time()
    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, 
                                                           HRIR_R, 
                                                           headRot, 
                                                           soundSource[0], 
                                                           NumElePosition)
    end = time.time()
    print("interpolation: ", end-start)

def schedule_update_HRIR():
    global current_task
    
    if current_task:
        current_task.cancel()
    
    current_task = asyncio.create_task(update_HRIR())
    
async def processing_audio_async(instrumentIndex, i):
    global HRIR_L_INT, HRIR_R_INT, AudioFile
    # start = time.time()
    j = instrumentIndex
    
    soundOutLArray[j], soundOutRArray[j], inBufferArray[j], outBufferLArray[j], outBufferRArray[j] = await HRTF_Effect.HRTFEffect_async(HRIR_L_INT, 
                                                                                                                                        HRIR_R_INT, 
                                                                                                                                        AudioFile[j][i*HopSize:(i+1)*HopSize], 
                                                                                                                                        inBufferArray[j], 
                                                                                                                                        outBufferLArray[j], 
                                                                                                                                        outBufferRArray[j])
    # end = time.time()
    # print("at: ", j, "process audio: ", end-start)
    
def processing_audio(instrumentIndex, i):
    global HRIR_L_INT, HRIR_R_INT, outBufferLArray, outBufferRArray, inBufferArray, soundOutLArray, soundOutRArray, AudioFile
    # start = time.time()
    j = instrumentIndex
    
    soundOutLArray[j], soundOutRArray[j], inBufferArray[j], outBufferLArray[j], outBufferRArray[j] = HRTF_Effect.HRTFEffect(HRIR_L_INT, 
                                                                                                                            HRIR_R_INT, 
                                                                                                                            AudioFile[j][i*HopSize:(i+1)*HopSize], 
                                                                                                                            inBufferArray[j], 
                                                                                                                            outBufferLArray[j], 
                                                                                                                            outBufferRArray[j])
    
    # end = time.time()
    # print("at: ", j, "process audio: ", end-start)
    
def processing_audio_multi(instrumentIndex, i, soundOutLArray, soundOutRArray, HRIR_L_INT, HRIR_R_INT):
    global outBufferLArray, outBufferRArray, inBufferArray, AudioFile
    # start = time.time()
    j = instrumentIndex
    
    soundOutLArray[j], soundOutRArray[j], inBufferArray[j], outBufferLArray[j], outBufferRArray[j] = HRTF_Effect.HRTFEffect(HRIR_L_INT, 
                                                                                                                            HRIR_R_INT, 
                                                                                                                            AudioFile[j][i*HopSize:(i+1)*HopSize], 
                                                                                                                            inBufferArray[j], 
                                                                                                                            outBufferLArray[j], 
                                                                                                                            outBufferRArray[j])

    # end = time.time()
    # print("at: ", j, "process audio: ", end-start)

async def single_HRTFEffect(i):
    global HRIR_L_INT, HRIR_R_INT, outBufferLArray, outBufferRArray, inBufferArray, soundOutLArray, soundOutRArray, AudioFile
    
    # 0: electric guitar
    # 1: piano
    # 2: vocal
    # 3: other instrunments
    # 4: drum
    # 5: bass
    instrumentIndex = 1
    processing_audio(instrumentIndex, i)
        
    left_list = np.array(soundOutLArray[instrumentIndex]).flatten().tolist()
    right_list = np.array(soundOutRArray[instrumentIndex]).flatten().tolist()

    json_data = {
        "index": i,
        "left": left_list,
        "right": right_list
    }
    
    return json_data

async def async_HRTFEffect(i):
    global soundOutLArray, soundOutRArray, AudioFile

    tasks = []
    for index in range(len(AudioFile)):
        task = asyncio.create_task(processing_audio_async(index, i))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    # start = time.time()
    _left_list = [sum(sum(x)) for x in zip(*soundOutLArray)]
    _right_list = [sum(sum(x)) for x in zip(*soundOutRArray)]
    # end = time.time()
    
    # print("adding: ", end-start)
    
    json_data = {
        "index": i,
        "left": _left_list,
        "right": _right_list
    }
    
    return json_data

def multi_thread_HRTFEffect(i):
    global soundOutLArray, soundOutRArray, AudioFile, executor_thread

    futures = [executor_thread.submit(processing_audio, index, i) for index in range(len(AudioFile))]

    for future in futures:
        future.result()

    # start = time.time()
    _left_list = [sum(sum(x)) for x in zip(*soundOutLArray)]
    _right_list = [sum(sum(x)) for x in zip(*soundOutRArray)]
    # end = time.time()

    # print("adding: ", end-start)

    json_data = {
        "index": i,
        "left": _left_list,
        "right": _right_list
    }

    return json_data

def multi_process_HRTFEffect(i):
    global soundOutLArray, soundOutRArray, AudioFile, executor_process, HRIR_L_INT, HRIR_R_INT

    with Manager() as manager:
        soundOutLArray = manager.list(soundOutLArray)
        soundOutRArray = manager.list(soundOutRArray)
        HRIR_L_INT = manager.list(HRIR_L_INT)
        HRIR_R_INT = manager.list(HRIR_R_INT)

        futures = [executor_process.submit(processing_audio_multi, index, i, soundOutLArray, soundOutRArray, HRIR_L_INT, HRIR_R_INT) for index in range(len(AudioFile))]

        for future in futures:
            future.result()

        # start = time.time()
        _left_list = [sum(sum(x)) for x in zip(*soundOutLArray)]
        _right_list = [sum(sum(x)) for x in zip(*soundOutRArray)]
        # end = time.time()
            
        # print(_left_list[0:10])

        # print("adding: ", end-start)

        json_data = {
            "index": i,
            "left": _left_list,
            "right": _right_list
        }

        return json_data