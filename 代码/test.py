from definition import Song
import calculate
import numpy as np
from tool import *

calculate.Calculate()
music = calculate.regeneration()
list = np.zeros((1, 1))
for i in range(len(music)):
    print(Chord_Dig.get(music[i].chord))
