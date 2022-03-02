from musicpy import *
LC = [2, 1, 1]  # 和弦位置参数
LP = [2, 1, 2, 1, 2, 1, 2, 1]  # 旋律位置参数
Note_Dig = {1: 0, 12: 1, 2: 2, 23: 3, 3: 4, 4: 5,
            45: 6, 5: 7, 56: 8, 6: 9, 67: 10, 7: 11}  # 音符数字化
Chord_Dig = {'C': [0, 4, 7], 'Cm': [0, 3, 7],
             'D': [2, 6, 9], 'Dm': [2, 5, 9],
             'E': [4, 8, 11], 'Em': [4, 7, 11],
             'F': [5, 9, 0], 'Fm': [5, 8, 0],
             'G': [7, 11, 2], 'Gm': [7, 10, 2],
             'A': [9, 1, 4], 'Am': [9, 0, 4],
             'B': [11, 3, 6], 'Bm': [11, 2, 6]}  # 和弦数字化
Chord_Level = {'C': 1, 'Cm': 1, 'D': 2, 'Dm': 2, 'E': 3, 'Em': 3, 'F': 4,
               'Fm': 4, 'G': 5, 'Gm': 5, 'A': 6, 'Am': 6, 'B': 7, 'Bm': 7}  # 和弦级数
ChordSet = [[0, 3, 7], [0, 4, 7],
            [2, 5, 9], [2, 6, 9],
            [4, 7, 11], [4, 8, 11],
            [5, 8, 0], [5, 9, 0],
            [7, 10, 2], [7, 11, 2],
            [9, 0, 4], [9, 1, 4],
            [11, 2, 6], [11, 3, 6]]  # 和弦集
Level2Chord = {1: 'C', 2: 'Dm', 3: 'Em', 4: 'F', 5: 'G', 6: 'Am', 7: 'Bm'}  # 级数对应和弦
Dig2Note = {1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'A',
            7: 'B', 12: 'C#', 23: 'D#', 45: 'F#', 56: 'G#', 67: 'A#'}  # 编曲用


class Song:
    def __init__(self):
        self.name = ""  # 歌名
        self.singer = ""  # 歌手
        self.style = ""  # 风格
        self.tune = []  # 只记录部分副歌


def Play(song_tune):  # 播放tune
    default = 5  # 默认音高
    # https://wenku.baidu.com/view/ed23999bd5bbfd0a78567338.html 乐器对照表
    melody = song_tune
    num = len(melody)
    total = chord([])
    for i in range(num):
        bar = melody[i]
        cho = []
        for j in range(len(bar[2])):
            cho.append(Dig2Note[bar[1][j]] + str(default + bar[2][j]))
        total += chord(cho, interval=1 / 8, duration=1 / 8)
    play(total, bpm=100, instrument=76)  # 速度，乐器


def Tune_Init(songs):  # 将简谱数字化
    for song in songs:
        for t in song.tune:
            for i in range(8):
                t[1][i] = Note_Dig[t[1][i]]
    return songs


def Tune_Play(songs):
    for song in songs:
        print("正在播放：", song.name)
        Play(song.tune)
        time.sleep(20)


def HNN(c_note, p_note, hnn):  # Harmony of Chord_Note and Piece_Note
    s = (p_note - c_note + 96) % 12
    return hnn[s]


def HCN(chord, p_note, hnn):  # Harmony of Chord and Piece_Note
    s = 0
    for i in range(3):
        s += (LC[i] * HNN(chord[i], p_note, hnn))
    return s


def HCP(chord, piece, p, hnn):  # Harmony of Chord and Piece
    s = 0
    for i in range(8):
        s += (LP[i] * HCN(chord, piece[i], hnn)) ** p
    s = s ** (1 / p)
    return s


def RHCP(chord, piece, p, hnn):  # Range of HCP
    hcp_max = HCP(chord, piece, p, hnn)  # considered the chord as the best to the piece
    hcp_min = 1000
    for chord in ChordSet:
        temp = HCP(chord, piece, p, hnn)
        if temp < hcp_min:
            hcp_min = temp
    return hcp_max - hcp_min


def SRHCP(song, p, hnn):  # RHCP of the whole song
    s = 0
    for t in song.tune:
        s += RHCP(Chord_Dig[t[0]], t[1], p, hnn)
    return s


def ASRHCP(songs, p, hnn):  # SRHCP of all the songs
    s = 0
    for song in songs:
        s += SRHCP(song, p, hnn)
    return s


def Norm(x, p):  # p-norm of x
    s = 0
    lenth = len(x)
    for i in range(lenth):
        s = s + (x[i]) ** p
    return s ** (1 / p)
