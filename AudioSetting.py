import HRTF_Effect
import time
import numpy as np
import sounddevice as sd
from multiprocessing import Pool 
from functools import partial
from dataSet import indexNum, HopSize, AudioFile, headRot, soundSource, NumElePosition, HRIR_L, HRIR_R, FS, FrameSize, fileList

indexLength = 2000


def mainProcessing(indexNum, HopSize, AudioFile, headRot, soundSource, NumElePosition, HRIR_L, HRIR_R, FS, FrameSize, j):

    ind = 0
    soundOutLArray = []
    soundOutRArray = []

    Music = np.zeros((indexNum, HopSize))
    for i in range(0, indexNum-1):
        Music[i] = AudioFile[j, HopSize*i:HopSize*(i+1)]
    
    ## Buffer setup
    outBufferNum = FS + FrameSize
    outBufferL= np.zeros((outBufferNum, 1))
    outBufferR = np.zeros((outBufferNum, 1)) 

    inBufferNum = FrameSize
    inBuffer = np.zeros(inBufferNum)

    ## Source Position

    HRIR_L_INT, HRIR_R_INT = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource[j, :], NumElePosition)

    while ind < indexLength:

        soundOutL, soundOutR, inBuffer, outBufferL, outBufferR = HRTF_Effect.HRTFEffect(HRIR_L_INT, HRIR_R_INT, Music[ind], inBuffer, outBufferL, outBufferR)
        soundOutLArray = np.append(soundOutLArray, soundOutL) ## 모아놓은 data 한 번에
        soundOutRArray = np.append(soundOutRArray, soundOutR) ## 모아놓은 data 한 번에

        ind = ind+1

    AudioL = soundOutLArray
    AudioR = soundOutRArray

    Audio = np.vstack((AudioL, AudioR))
    Audio = np.transpose(Audio)
    
    return Audio


if __name__ == "__main__":
    num_cores = len(fileList)
    pool = Pool(num_cores)
    
    start = time.time()
    fMainProcessing = partial(mainProcessing, indexNum, HopSize, AudioFile, headRot, soundSource, NumElePosition, HRIR_L, HRIR_R, FS, FrameSize)
    output = pool.map(fMainProcessing, range(0, len(fileList)))
    end = time.time()

    print(end-start)
    OUTPUT = np.asarray(output)

    OUT = np.zeros((indexLength*HopSize, 2))
    for ii in range(0, len(fileList)):
        OUT = OUT + OUTPUT[ii]

    sd.play(OUT, FS)
    sd.wait()