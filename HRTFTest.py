import pandas as pd
import librosa
import scipy
import HRTF_Effect
from scipy import signal
import time

## HRIR Data
HRIR_L = pd.read_csv('Source/csv/HRIR_L.csv')
HRIR_R = pd.read_csv('Source/csv/HRIR_R.csv')

FS = 44100
## Music data
AudioFile, FSAudio = librosa.load('Source/csv/soundhelix_mono.mp3')
SamplingNum = int(AudioFile.size * FS / FSAudio)
AudioFile = scipy.signal.resample(AudioFile, SamplingNum) ## Mono Audiofile

## Head Rotation Data = Yaw/ Pitch/ Roll (Deg), Sound source Data = Azimuth/ Elevation (Deg)
headRot = [0, 0, 0]
soundSource = [270, 30]

start = time.time()
HRIR_L, HRIR_R = HRTF_Effect.InterpolationHRIR(HRIR_L, HRIR_R, headRot, soundSource)
end = time.time()
print(end-start)

HRTF_Effect.HRTFEffect(HRIR_L, HRIR_R, AudioFile)