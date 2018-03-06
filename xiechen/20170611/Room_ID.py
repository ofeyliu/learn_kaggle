# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 20:34:16 2017

@author: yli
"""
import os
import numpy as np

path = 'train_data.txt'
save_path = 'room_id.txt'
num = 10000

if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_roomid(datapath):
    f= open(datapath,'r')
    data = f.readlines()
    room_id = [x.split()[5] for x in data]
    return room_id[1:]
    
room_id = get_roomid(path)
    
    
