import numpy as np
from scipy.fft import fft, ifft
import pandas as pd
import math
import time
import librosa
import scipy

## Interpolation Data
lsGroups = np.array(pd.read_csv('Source/csv/ls_groups.csv'))
posXYZ =  np.array(pd.read_csv('Source/csv/SourcePositionXYZ.csv'))
invmtx =  np.array(pd.read_csv('Source/csv/invmtx.csv'))

FrameSize = 1024
HopSize = int(FrameSize/2)
AWin = np.hanning(FrameSize)

def AudioRead(fileName, FS):
    AudioFile, FSAudio = librosa.load(fileName)
    SamplingNum = int(AudioFile.size * FS / FSAudio)
    AudioFile = scipy.signal.resample(AudioFile, SamplingNum) ## Mono Audiofile

    return AudioFile

def cart2sph(x, y, z):

    radius = np.sqrt(x**2 + y**2 + z**2)  # radius
    azimuth = np.arctan2(y, x)  # azimuth angle
    elevation = np.arccos(z / radius)  # elevation angle

    return azimuth, elevation, radius

def sph2cart(azimuth, elevation, r):

    x = r * math.sin(elevation) * math.cos(azimuth)
    y = r * math.sin(elevation) * math.sin(azimuth)
    z = r * math.cos(elevation)

    return x, y, z

def pov2sph(pov, sourcePosition):

    xyzSource = sph2cart(np.radians(sourcePosition[0]), np.radians(sourcePosition[1]), 1)
    dvec = np.transpose(xyzSource)

    yaw = np.radians(pov[0])
    pitch = np.radians(pov[1])
    roll = np.radians(pov[2])
    
    RotYaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    RotPitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    RotRoll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    Rot = np.dot(RotYaw, RotPitch, RotRoll)
    dvecr = np.dot(Rot, dvec[:])
    az, el, r = cart2sph(dvecr[0], dvecr[1], dvecr[2])
    
    return az, el

def triIntersect(T, v):
    dang = np.dot(v, T)
    
    if any(dang < 0):
        btype = 0
    else:
        d = np.zeros(3)
        d[0] = np.linalg.det(np.column_stack((T[:, [0, 1]], v)))
        d[1] = np.linalg.det(np.column_stack((T[:, [0]], v, T[:, [2]])))
        d[2] = np.linalg.det(np.column_stack((v, T[:, [1]], T[:, [2]])))

        sd = np.sign(d)
        
        if sum(abs(d) < 1e-12) > 1: # crossing with vertex
            btype = 3
        elif abs(sum(sd)) > 1: # edge crossing or inside
            if sum(abs(d) < 1e-12) > 0:
                btype = 2
            else:
                btype = 1
        else:
            btype = 0
    
    return btype

def divideSourcePostion(SourcePosition):
    NumElePosition = np.zeros(19)
    for i in range(0, len(SourcePosition)):
        SP = -SourcePosition[i, 1]
        index = int(SP/10 + 9)
        NumElePosition[index] += 1
    return NumElePosition

def InterpolationHRIR(HRIR_L, HRIR_R, pov, sourcePosition, NumElePosition):
    global lsGroups, posXYZ, invmtx
    
    ## 머리 각도와 sound source의 각도를 모두 고려한 각도 DATA
    relatedSph = pov2sph(pov, sourcePosition)
    relatedEleDegree = relatedSph[1]*180/math.pi
    NumMin = 0
    NumMax = 0

    if relatedEleDegree >= 0 and relatedEleDegree < 90:
        relatedEleDegree = 90-relatedEleDegree
        eleIndex = int(relatedEleDegree//10 + 9)

        for ii in range(eleIndex-1, 19):
            NumMax += int(NumElePosition[ii])
        for jj in range(eleIndex+3, 19):
            NumMin += int(NumElePosition[jj])
    
    elif  relatedEleDegree > 90 and relatedEleDegree <= 180:
        relatedEleDegree = 90-relatedEleDegree
        eleIndex = int(relatedEleDegree//10 + 9)

        for ii in range(eleIndex-1, 19):
            NumMax += int(NumElePosition[ii])
        for jj in range(eleIndex+3, 19):
            NumMin += int(NumElePosition[jj])

    else:
        for ii in range(9, 19):
            NumMax += int(NumElePosition[ii])
        for jj in range(11, 19):
            NumMin += int(NumElePosition[jj])

    q = sph2cart(relatedSph[0], relatedSph[1], 1)

    inters = np.zeros(len(lsGroups))
    NumMax = NumMax+2

    for i in range(0, len(lsGroups)):
        LS = lsGroups[i, :]
        if min(LS) >= NumMin and max(LS) <= NumMax:
            # print(LS)
            pos = np.zeros((LS.size, LS.size))

            for ii in range (0, LS.size):
                pos[ii, :] = posXYZ[(LS[ii]-1), :]

            posTranspose = np.transpose(pos)
            # pos = posXYZ[lsGroups[i, :], :]
            inters[i] = triIntersect(posTranspose, q)

    idx = np.argwhere(inters)
    idxFind = int(idx[0])

    gGainsF = np.reshape(invmtx[idxFind, :], [3, 3])
    gGainsFTranspose = np.transpose(gGainsF)
    gGainsFCir = np.dot(gGainsFTranspose, q)

    gGains = gGainsFCir/sum(gGainsFCir)
    
    HRIR_L = np.dot(gGains, HRIR_L[lsGroups[idxFind, :], :])
    HRIR_R = np.dot(gGains, HRIR_R[lsGroups[idxFind, :], :])

    return HRIR_L, HRIR_R

def fftfilt(b, x):
    nfft = 2 ** np.ceil(np.log2(len(x)+len(b)-1)).astype(int)

    X = fft(x, n=nfft)
    B = fft(b, n=nfft)

    Y = X*B
    
    y = np.real(ifft(Y))
    
    return y[:len(x)]    


def HRTFEffect(HRIR_L, HRIR_R, AudioFile, inBuffer, outBufferL, outBufferR):
    global FrameSize, HopSize, AWin
    
    inBuffer = np.append(inBuffer[HopSize:], AudioFile, axis = 0)
    dataIn = inBuffer * AWin
    
    dataHRTF_L = fftfilt(HRIR_L, dataIn)
    dataHRTF_R = fftfilt(HRIR_R, dataIn)
    
    for j in range(0, dataHRTF_L.size):
        outBufferL[j] = outBufferL[j]+dataHRTF_L[j]
        outBufferR[j] = outBufferR[j]+dataHRTF_R[j]
    
    soundOutL = outBufferL[0:HopSize, :] ## Buffer당 output
    soundOutR = outBufferR[0:HopSize, :]

    outBufferL = np.append(outBufferL[HopSize:, :], np.zeros((HopSize, 1)), axis = 0)
    outBufferR = np.append(outBufferR[HopSize:, :], np.zeros((HopSize, 1)), axis = 0)

    return soundOutL, soundOutR, inBuffer, outBufferL, outBufferR

async def HRTFEffect_async(HRIR_L, HRIR_R, AudioFile, inBuffer, outBufferL, outBufferR):
    global FrameSize, HopSize, AWin

    inBuffer = np.append(inBuffer[HopSize:], AudioFile, axis = 0)
    dataIn = inBuffer * AWin
    
    dataHRTF_L = fftfilt(HRIR_L, dataIn)
    dataHRTF_R = fftfilt(HRIR_R, dataIn)
    
    for j in range(0, dataHRTF_L.size):
        outBufferL[j] = outBufferL[j]+dataHRTF_L[j]
        outBufferR[j] = outBufferR[j]+dataHRTF_R[j]
    
    soundOutL = outBufferL[0:HopSize, :] ## Buffer당 output
    soundOutR = outBufferR[0:HopSize, :]

    outBufferL = np.append(outBufferL[HopSize:, :], np.zeros((HopSize, 1)), axis = 0)
    outBufferR = np.append(outBufferR[HopSize:, :], np.zeros((HopSize, 1)), axis = 0)

    return soundOutL, soundOutR, inBuffer, outBufferL, outBufferR


def cocktail(music, targetMusic):
    start = time.time()
    musicFFT = fft(music)
    end = time.time()
    
    return end-start