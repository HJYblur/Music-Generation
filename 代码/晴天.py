import pickle
import os

class Song:
    def __init__(self):
        self.name=""#歌名
        self.singer=""#歌手
        self.style=""#风格
        self.tune=[]#只记录部分副歌

song=Song()
song.name="晴天"
song.singer={"周杰伦"}
song.style={}
song.tune=[
    ['C',
     [3,2,4,3,3,1,5,7],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [1,7,5,1,1,1,6,6],
     [1,0,0,0,0,0,0,0]],
    ['F',
     [6,6,5,5,5,5,4,3],
     [0,0,0,0,0,0,0,0]],
    ['C',
     [2,3,4,3,3,3,3,3],
     [0,0,0,0,0,0,0,0]],
    ['E',
     [3,45,56,3,3,45,56,7],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [2,7,1,1,1,1,1,1],
     [1,0,1,1,1,1,1,1]],
    ['F',
     [1,5,5,6,5,4,2,3],
     [1,0,0,0,0,0,0,0]],
    ['G',
     [4,5,6,1,6,7,7,7],
     [0,0,0,0,0,0,0,0]],
    ['C',
     [3,2,4,3,3,1,5,7],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [1,7,5,1,1,1,6,6],
     [1,0,0,0,0,0,0,0]],
    ['F',
     [6,6,5,5,5,5,4,3],
     [0,0,0,0,0,0,0,0]],
    ['C',
     [2,3,4,3,3,3,3,3],
     [0,0,0,0,0,0,0,0]],
    ['E',
     [3,45,56,3,3,45,56,7],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [2,7,1,1,1,1,1,1],
     [1,0,1,1,1,1,1,1]],
    ['F',
     [1,5,5,6,5,4,6,7],
     [1,0,0,0,0,0,-1,-1]],
    ['G',
     [1,2,3,2,2,2,3,1],
     [0,0,0,0,0,0,0,0]],
    ['C',
     [1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0]]]

file=open('晴天.pkl','wb')
pickle.dump(song,file)
file.close()
