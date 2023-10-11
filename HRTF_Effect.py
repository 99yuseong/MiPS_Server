import numpy as np
from scipy.fft import fft, ifft
import sounddevice as sd
import pandas as pd
import math
import time

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

def InterpolationHRIR(HRIR_L, HRIR_R, pov, sourcePosition):
    
    ## 머리 각도와 sound source의 각도를 모두 고려한 각도 DATA
    relatedSph = pov2sph(pov, sourcePosition)
    q = sph2cart(relatedSph[0], relatedSph[1], 1)

    lsGroups = pd.read_csv('/Users/kyoungyeongo/Documents/MiPS/ls_groups.csv')
    lsGroups = np.array(lsGroups)
    inters = np.zeros(len(lsGroups))
    posXYZ = pd.read_csv('/Users/kyoungyeongo/Documents/MiPS/SourcePositionXYZ.csv')
    posXYZ = np.array(posXYZ)

    HRIR_L = np.array(HRIR_L)
    HRIR_R = np.array(HRIR_R)

    for i in range(0, len(lsGroups)):
        LS = lsGroups[i, :]
        # print(LS)
        pos = np.zeros((LS.size, LS.size))

        for ii in range (0, LS.size):
            pos[ii, :] = posXYZ[(LS[ii]-1), :]

        posTranspose = np.transpose(pos)
        # pos = posXYZ[lsGroups[i, :], :]
        inters[i] = triIntersect(posTranspose, q)

    idx = np.argwhere(inters)
    idxFind = int(idx[0])

    invmtx = pd.read_csv('/Users/kyoungyeongo/Documents/MiPS/invmtx.csv')
    invmtx = np.array(invmtx)
    gGainsF = np.reshape(invmtx[idxFind, :], [3, 3])
    gGainsFTranspose = np.transpose(gGainsF)
    gGainsFCir = np.dot(gGainsFTranspose, q)

    gGains = gGainsFCir/sum(gGainsFCir)
    
    HRIR_L = np.dot(gGains, HRIR_L[lsGroups[idxFind, :], :])
    HRIR_R = np.dot(gGains, HRIR_R[lsGroups[idxFind, :], :])

    return HRIR_L, HRIR_R

def fftfilt(b, x):
    b = np.array(b)
    nfft = 2 ** np.ceil(np.log2(len(x)+len(b)-1)).astype(int)

    X = fft(x, n=nfft)
    B = fft(b, n=nfft)

    Y = X*B
    
    y = np.real(ifft(Y))
    
    return y[:len(x)]    


def HRTFEffect(HRIR_L, HRIR_R, AudioFile):
    
    ## Setup
    FrameSize = 1024
    FS = 44100
    HopSize = int(FrameSize/2)
    NHop = FrameSize/HopSize
    AWin = np.hanning(FrameSize)

    if AudioFile.size % HopSize != 0:
        remain = HopSize - AudioFile.size % HopSize
        AudioFile = np.append(AudioFile, np.zeros(remain), axis = 0)
        
    indexNum = int(AudioFile.size/HopSize)

    Music = np.zeros((indexNum, HopSize))
    for i in range(0, indexNum-1):
        Music[i] = AudioFile[HopSize*i:HopSize*(i+1)]

    ## Buffer setup
    outBufferNum = FS + FrameSize
    outBuffer = np.zeros((outBufferNum, 2))

    inBufferNum = FrameSize
    inBuffer = np.zeros(inBufferNum)
        
    ind = 0
    soundOutArray = np.empty((0, 2))

    # start = time.time()
    while ind < 100:
    # while ind < indexNum:

        inBuffer = np.append(inBuffer[HopSize:], Music[ind], axis = 0)
        dataIn = inBuffer * AWin
        
        dataHRTF_L = fftfilt(HRIR_L, dataIn)
        dataHRTF_R = fftfilt(HRIR_R, dataIn)
        
        for j in range(0, dataHRTF_L.size):
            outBuffer[j, 0] = outBuffer[j, 0]+dataHRTF_L[j]
            outBuffer[j, 1] = outBuffer[j, 1]+dataHRTF_R[j]
        
        soundOut = outBuffer[0:HopSize, :] ## Buffer당 output

        soundOutArray = np.append(soundOutArray, soundOut, axis = 0) ## 모아놓은 data 한 번에
        
        outBuffer = np.append(outBuffer[HopSize:, :], np.zeros((HopSize, 2)), axis = 0)
        ind = ind + 1
    # end = time.time()
    # print(end-start)

    sd.play(soundOutArray, FS)
    sd.wait()