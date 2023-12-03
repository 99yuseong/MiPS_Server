import pandas as pd
import HRTF_Effect
from scipy import signal
import numpy as np
import os

FrameSize = 1024
FS = 44100
HopSize = int(FrameSize/2)

## HRIR Data
HRIR_L = np.array(pd.read_csv('Source/csv/HRIR_L.csv'))
HRIR_R = np.array(pd.read_csv('Source/csv/HRIR_R.csv'))

SourcePosition = np.array(pd.read_csv('Source/csv/SourcePosition.csv'))
NumElePosition = HRTF_Effect.divideSourcePostion(SourcePosition)

## Music data
fileDir = 'Source/Music/'
fileList = os.listdir(fileDir)

InstrunmentList = []

for i in range(0, len(fileList)):
    AudioData = HRTF_Effect.AudioRead(fileDir+fileList[i], FS)
    # AudioFile = np.concatenate((AudioFile, AudioData), axis = 1)
    if i == 0:
        AudioFile = AudioData
        #  originGain = AudioFile[0]
    else:
        AudioFile = np.vstack((AudioFile, AudioData))
        # originGain = originGain + AudioFile[i][0]

    InstrunmentList = np.append(InstrunmentList, fileList[i].split('.')[0])

outBufferLArray = [np.zeros((FS + FrameSize, 1)) for _ in range(len(fileList))]
outBufferRArray = [np.zeros((FS + FrameSize, 1)) for _ in range(len(fileList))]
inBufferArray = [np.zeros(FrameSize) for _ in range(len(fileList))]
soundOutLArray = [np.zeros((HopSize, 1)) for _ in range(len(fileList))]
soundOutRArray = [np.zeros((HopSize, 1)) for _ in range(len(fileList))]

time = HRTF_Effect.cocktail(AudioFile[1, :], AudioFile[2, :])

if len(AudioFile[0, :]) % HopSize != 0:
    remain = HopSize - len(AudioFile) % HopSize
AudioFile = np.concatenate((AudioFile, np.zeros((len(fileList), remain))), axis = 1)

indexNum = int(len(AudioFile[0, :])/HopSize)

## Head Rotation Data = Yaw/ Pitch/ Roll (Deg), Sound source Data = Azimuth/ Elevation (Deg)
headRot = [0, 0, 0]
soundSource = np.empty((len(fileList), 2))

soundSource[0, 0] = 0       
soundSource[0, 1] = 0      # electric guitar
soundSource[1, 0] = 0
soundSource[1, 1] = 0      # piano
soundSource[2, 0] = 0
soundSource[2, 1] = 0      # vocal
# soundSource[3, 0] = 0
# soundSource[3, 1] = 0      # other instrunments
# soundSource[4, 0] = 0
# soundSource[4, 1] = 90      # drum
# soundSource[5, 0] = 0
# soundSource[5, 1] = 90      # bass