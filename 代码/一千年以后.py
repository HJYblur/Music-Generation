import pickle
import os

class Song:
    def __init__(self):
        self.name=""#歌名
        self.singer=""#歌手
        self.style=""#风格
        self.tune=[]#只记录部分副歌

song=Song()
song.name="一千年以后"
song.singer={"林俊杰"}
song.style={}
song.tune=[
    ['C',
     [3,3,2,1,1,1,5,5],
     [0,0,0,0,0,0,-1,-1]],
    ['G',
     [2,2,2,2,2,2,3,2],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [1,1,7,6,6,6,3,3,],
     [0,0,-1,-1,-1,-1,-1,-1]],
    ['G',
     [7,7,7,7,7,7,1,7],
     [-1,-1,-1,-1,-1,-1,0,-1]],
    ['F',
     [6,6,7,1,1,1,2,2],
     [-1,-1,-1,0,0,0,0,0]],
    ['C',
     [5,5,3,3,3,3,3,3],
     [-1,-1,0,0,0,0,0,0]],
    ['Dm',
     [4,4,3,1,1,1,3,3],
     [0,0,0,0,0,0,0,0]],
    ['G',
     [2,2,2,2,2,7,1,2],
     [0,0,0,0,0,-1,0,0]],
    ['C',
     [3,3,2,1,1,1,5,5],
     [0,0,0,0,0,0,0,0]],
    ['G',
     [3,3,2,2,2,2,3,2],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [1,1,7,6,6,6,2,7,],
     [0,0,-1,-1,-1,-1,0,-1]],
    ['G',
     [7,7,7,7,7,7,1,7],
     [-1,-1,-1,-1,-1,-1,0,-1]],
    ['F',
     [6,6,5,6,6,6,1,1],
     [-1,-1,-1,-1,-1,-1,0,0]],
    ['C',
     [5,5,2,1,1,1,3,3],
     [-1,-1,0,0,0,0,0,0]],
    ['Dm',
     [4,4,3,2,2,2,1,2],
     [0,0,0,0,0,0,0,0]],
    ['Fm',
     [23,23,2,1,23,23,2,1],
     [0,0,0,0,0,0,0,0]],
    ['C',
     [5,5,1,1,1,1,1,1],
     [-1,-1,0,0,0,0,0,0]]]

file=open('一千年以后.pkl','wb')
pickle.dump(song,file)
file.close()
