# -*- coding: utf-8 -*-
import socket
import sys
import cv2
import pickle

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
running = 1
size = 100000
host = "10.200.185.234"
port = int(6667)
path = 'test.jpg'
#s.setdefaulttimeout(600.00)
s.connect((host,port))
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#RGB numpy array
img = cv2.resize(img, (112,112), interpolation = cv2.INTER_AREA)
data_string = pickle.dumps(img, protocol=2)
s.send(data_string)
data = s.recv(size)
print(data)
s.close()