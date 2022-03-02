import pickle
import os

class Song:
    def __init__(self):
        self.name=""
        self.singer=""
        self.style=""
        self.tune=[]

song=Song()
song.name="万有引力"
song.singer={"汪苏泷"}
song.style={}
song.tune=[
    ['C',
     [3,3,3,3,3,3,4,5],
     [0,0,0,0,0,0,0,0]],
    ['G',
     [5,2,2,2,2,3,2,1],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [1,1,1,1,1,2,3,7],
     [0,0,0,0,0,0,0,0]],
    ['Em',
     [7,6,6,5,5,3,3,5],
     [0,0,0,0,0,0,0,0]],
    ['F',
     [6,1,1,1,1,7,1,5],
     [0,0,0,0,0,-1,0,0]],
    ['Em',
     [5,1,7,1,1,2,3,4],
     [0,0,-1,0,0,0,0,0]],
    ['Dm',
     [4,3,4,4,4,5,3,2],
     [0,0,0,0,0,0,0,0]],
    ['G',
     [2,2,2,5,1,2,3,3],
     [0,0,0,-1,0,0,0,0]],
    ['C',
     [3,3,3,3,3,3,4,5],
     [0,0,0,0,0,0,0,0]],
    ['G',
     [5,2,2,2,2,3,2,1],
     [0,0,0,0,0,0,0,0]],
    ['Am',
     [1,1,1,1,1,2,3,7],
     [0,0,0,0,0,0,0,0]],
    ['Em',
     [7,6,6,5,5,3,3,5],
     [0,0,0,0,0,0,0,0]],
    ['F',
     [6,1,1,1,1,7,1,5],
     [0,0,0,0,0,-1,0,0]],
    ['Em',
     [5,1,7,1,1,1,1,5],
     [0,0,-1,0,0,0,0,-1]],
    ['Dm',
     [4,3,1,1,1,2,3,2],
     [0,0,0,0,0,0,0,0]],
    ['G',
     [2,2,2,5,3,5,2,1],
     [0,0,0,-1,0,-1,0,0]],
    ['C',
     [1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0]]]

file=open('万有引力.pkl','wb')
pickle.dump(song,file)
file.close()
