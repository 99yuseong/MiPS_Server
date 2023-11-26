import pandas as pd
import HRTF_Effect
from scipy import signal
import numpy as np
import os

## HRIR Data
HRIR_L = pd.read_csv('Source/csv/HRIR_L.csv')
HRIR_R = pd.read_csv('Source/csv/HRIR_R.csv')

SourcePosition = pd.read_csv('Source/csv/SourcePosition.csv')
NumElePosition = HRTF_Effect.divideSourcePostion(SourcePosition)

FS = 44100
## Music data
fileDir = 'Source/Music/'
fileList = os.listdir(fileDir)
print(fileList)
# fileList = np.delete(fileList, 0, axis = 0)
print(fileList)
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

time = HRTF_Effect.cocktail(AudioFile[1, :], AudioFile[2, :])
print(time)

FrameSize = 1024
HopSize = int(FrameSize/2)

if len(AudioFile[0, :]) % HopSize != 0:
        remain = HopSize - len(AudioFile) % HopSize
AudioFile = np.concatenate((AudioFile, np.zeros((len(fileList), remain))), axis = 1)

indexNum = int(len(AudioFile[0, :])/HopSize)

MusicArray = []
AudioL = []
AudioR = []
soundOutArray = []

## Head Rotation Data = Yaw/ Pitch/ Roll (Deg), Sound source Data = Azimuth/ Elevation (Deg)
headRot = [20, 10, 60]
soundSource = np.empty((len(fileList), 2))
print(soundSource.shape)
print(len(fileList))
soundSource[0, 0] = 0       
soundSource[0, 1] = 90      # electric guitar
soundSource[1, 0] = 0
soundSource[1, 1] = 90      # piano
soundSource[2, 0] = 300
soundSource[2, 1] = 90      # vocal
soundSource[3, 0] = 0
soundSource[3, 1] = 90      # other instrunments
soundSource[4, 0] = 0
soundSource[4, 1] = 90      # drum
soundSource[5, 0] = 0
soundSource[5, 1] = 90      # bass