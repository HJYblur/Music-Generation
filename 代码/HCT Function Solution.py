from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np
import pickle
import os
import random
import math


alpha=[2,1,1]#和弦和谐度参数
beta=[4,2,4,2,4,2,4,2]#小节和谐度参数
sev2twl={1:0,12:1,2:2,23:3,
         3:4,4:5,45:6,5:7,
         56:8,6:9,67:10,7:11}#简谱记法改为五线谱记法
cho2twl={'C':[0,4,7],'Cm':[0,3,7],
         'D':[2,6,9],'Dm':[2,5,9],
         'E':[4,8,11],'Em':[4,7,11],
         'F':[5,9,0],'Fm':[5,8,0],
         'G':[7,11,2],'Gm':[7,10,2],
         'A':[9,1,4],'Am':[9,0,4],
         'B':[11,3,6],'Bm':[11,2,6]}
CSET=[[0,3,7],[0,4,7],
      [2,5,9],[2,6,9],
      [4,7,11],[4,8,11],
      [5,8,0],[5,9,0],
      [7,10,2],[7,11,2],
      [9,0,4],[9,1,4],
      [11,2,6],[11,3,6]]#和弦集
#CSET=[[0,4,7],[2,6,9],[4,7,11],[4,8,11],[5,8,0],[5,9,0],[7,11,2],[9,0,4],[11,2,6]]
songs=[]

class Song:
    def __init__(self):
        self.name=""#歌名
        self.singer=""#歌手
        self.style=""#风格
        self.tune=[]#只记录部分副歌

def Tune_Init(songs):
    for song in songs:
        for t in song.tune:
           for i in range(8):
               t[1][i]=sev2twl[t[1][i]]
    return songs

def HNN(c,t,h):#音音和谐度函数
    s=(t-c+96)%12
    return h[s]

def HCN(C,t,h):
    s=0
    for i in range(3):
        hcn=alpha[i]*HNN(C[i],t,h)
        s+=hcn
    return s

def HCT(C,T,p,h):
    s=0
    for i in range(8):
            #s+=(h[12+j]*h[15+i]*HNN(C[j],T[i],h))**p
            s+=(beta[i]*HCN(C,T[i],h))**p
    s=s**(1/p)
    return s

def DHCT(C,T,p,h):
    hct=HCT(C,T,p,h)
    s=500
    for Ci in CSET:
        hct1=HCT(Ci,T,p,h)
        if hct1<s:
            s=hct1
    s=s-hct
    return s

def SDHCT(song,p,h):
    s=0
    for t in song.tune:
        s+=DHCT(cho2twl[t[0]],t[1],p,h)
    return s

def SDHCT_Sum(songs,p,h):
    s=0
    for song in songs:
        s+=SDHCT(song,p,h)
    return s

def h_sum(h):
    s=0
    for i in range(12):
        s=s+(h[i])*1
    return s**(1)

for i in os.listdir('song'):
    file=open('song'+os.sep+i,'rb')
    song=pickle.load(file)
    file.close()
    songs.append(song)

songs=Tune_Init(songs)
fun=lambda h:SDHCT_Sum(songs,3,h)#lambda x: max(x[1],x[0])#设置最优化函数
#fun=lambda h:DHCT(cho2twl['F'],[9,9,9,9,9,9,9,0],2,h)
#bounds=Bounds([1]*23,[2]*12+[4]*11)#边界条件
bounds=Bounds([1]*12,[10000]*12)#边界条件
cons=({'type':'eq','fun':lambda h:h_sum(h)-500})#约束条件
#cons=()


#h0=[random.random()+1 for i in range(12)]+[random.random()+2 for i in range(11)]#h,alpha,beta
h0=[random.random()*9999+1 for i in range(12)]#h,alpha,beta
#求解#
res = minimize(fun,h0,method='SLSQP',bounds=bounds,constraints=cons)
#####
print('最小值：',res.fun)
print('最优解：',res.x)
print('是否顺利进行：',res.success)
print('迭代终止原因：',res.message)

s=0
x=np.zeros(12)
for i in range(12):
    s+=res.x[i]**2
s=s**(1/2)
for i in range(12):
    x[i]=res.x[i]/s
print(x)
