import numpy as np
from definition import Song, Music
from read import songs
from tool import *

debug = False
Num = 10  # 生成的旋律数

# 7:7种和弦，4:0、2、4、6这4个位置，37:37种单音
stressedDic = np.zeros((7, 4, 37), dtype=int)
# 从左到右分别是：轻音，轻音前后的两个重音
unstressedDic = np.zeros((37, 37, 37), dtype=int)
hnn = [0.51455546, 0.07879776, 0.17814136, 0.36722034, 0.33739019, 0.27546282,
       0.03413839, 0.43721854, 0.26734545, 0.2695031, 0.16808418, 0.08968431]


def chord2num(chord):
    if chord == 'C' or chord == 'Cm':
        return 1
    elif chord == 'D' or chord == 'Dm':
        return 2
    elif chord == 'E' or chord == 'Em':
        return 3
    elif chord == 'F' or chord == 'Fm':
        return 4
    elif chord == 'G' or chord == 'Gm':
        return 5
    elif chord == 'A' or chord == 'Am':
        return 6
    elif chord == 'B' or chord == 'Bm':
        return 7
    else:
        return 0


def num2chord(n):
    if n == 1:
        return 'C'
    elif n == 2:
        return 'Dm'
    elif n == 3:
        return 'Em'
    elif n == 4:
        return 'F'
    elif n == 5:
        return 'G'
    elif n == 6:
        return 'Am'
    elif n == 7:
        return 'Bm'
    else:
        print('error')


def Calculate():
    for song in songs:
        # print(song.name)
        # print(song.tune)
        for i in range(len(song.tune)):
            tune = song.tune[i]
            chord = chord2num(tune[0]) - 1
            # print(tune[0], pos)
            if chord == -1:
                if debug:
                    print("We've got wrong chord:", tune[0])
                continue
            tuneList = []
            for i in range(7):
                tuneList.append(tune[1][i] + tune[2][i] * 12 + 12)
            # print(tuneList)
            for j in range(len(tuneList)):
                if j % 2 == 0:  # 重音位置
                    pos = round(j / 2)
                    syllable = tuneList[j]
                    stressedDic[chord, pos, syllable] += 1
                else:  # 轻音位置
                    front = tuneList[j - 1]
                    syllable = tuneList[j]
                    rear = 0
                    if j == 7:
                        if i == len(song.tune) - 1:
                            rear = 36
                        else:
                            rear = song.tune[i + 1, 1, 0]
                    else:
                        rear = tuneList[j + 1]
                    unstressedDic[syllable, front, rear] += 1

    if debug:
        np.set_printoptions(formatter={'int': '{:d}'.format})
        print("Below: StressedDic")
        print(stressedDic)
        print("Below: UnStressedDic")
        print(unstressedDic)


def sigmoid(X):
    return 1.0 / (1 + np.exp(-float(X)))


def normalization1(Dic):  # 计算概率
    dic = np.array(Dic, np.float32)
    if np.sum(dic[:]) != 0.0:
        dic[:] = pow(dic[:], 2)
        dic[:] = dic[:] / np.sum(dic[:])
    else:
        n = dic.shape[0]
        for i in range(n):
            dic[i] = 1 / n
    return dic


# def normalization2(Dic):  # 计算概率
#     dic = np.array(Dic, np.float32)
#     if np.sum(dic[:, :]) != 0.0:
#         dic[:, :] = dic[:, :] / np.sum(dic[:, :])
#     return dic


def normalization3(Dic):  # 计算概率
    dic = np.array(Dic, np.float32)
    if np.sum(dic[:, :, :]) != 0.0:
        dic[:, :, :] = pow(dic[:, :, :], 2)
        dic[:, :, :] = dic[:, :, :] / np.sum(dic[:, :, :])
    # print(dic)
    if debug:
        print("Below is the prediction of dic after normalization")
        np.set_printoptions(formatter={'float': '{:3f}'.format})
        print(dic)
    return dic


twlChangeList = {0: 1, 1: 12, 2: 2, 3: 23, 4: 3, 5: 4, 6: 45, 7: 5, 8: 56, 9: 6, 10: 67, 11: 7}


def twl2sev(twl):
    octave = []
    sev = []
    for i in range(twl.shape[0]):
        n = twl[i]
        octave.append(n // 12 - 1)
        sev.append(twlChangeList[n - (octave[i] + 1) * 12])
    return [sev, octave]


def regeneration():
    p1 = normalization3(stressedDic)
    for c in range(7):
        for i in range(4):
            for j in range(37):
                chord = num2chord(c + 1)
                list = Chord_Dig.get(chord)
                p1[c][i][j] *= HCN(list, j, hnn)
    p2 = normalization3(unstressedDic)

    # 生成和弦
    chordList = np.zeros((Num), dtype=int)
    pChord = np.zeros((7), dtype=float)
    for i in range(p1.shape[0]):
        pChord[i] = np.sum(p1[i, :, :])
    for i in range(Num):
        c = np.random.choice([1, 2, 3, 4, 5, 6, 7], p=normalization1(pChord).ravel())
        chordList[i] = c

    # 生成重音和轻音
    syllableList = np.zeros((Num, 8), dtype=int)
    pSyllable = np.zeros((37), dtype=float)
    for i in range(p1.shape[1]):
        for j in range(p1.shape[2]):
            pSyllable[j] = np.sum(p1[:, i, j])
        for k in range(Num):
            list = [x for x in range(37)]
            for iter in range(4):
                s = np.random.choice(list, p=normalization1(pSyllable).ravel())
                syllableList[k, iter * 2] = s
    for i in range(Num):
        for j in range(4):
            front = syllableList[i, j * 2]
            rear = 36
            if j != 3:
                rear = syllableList[i, j * 2 + 2]
            else:
                if i < Num - 1:
                    rear = syllableList[i + 1, 0]
            pUnstress = np.zeros((37), dtype=float)
            for k in range(p2.shape[0]):
                pUnstress[k] = p2[k, front, rear]
            s = np.random.choice(list, p=normalization1(pUnstress).ravel())
            syllableList[i, j * 2 + 1] = s
    music = []
    for i in range(Num):
        one = Music()
        one.chord = num2chord(chordList[i])
        [one.tune, one.octave] = twl2sev(syllableList[i])
        music.append(one)
    return music


def printMusic(dic):
    tune = []
    for i in range(len(dic)):
        print(dic[i].chord)
        np.set_printoptions(formatter={'int': '{:d}'.format})
        print(dic[i].tune)
        print(dic[i].octave)
        tune.append([dic[i].chord, dic[i].tune, dic[i].octave])
    Play(tune)
