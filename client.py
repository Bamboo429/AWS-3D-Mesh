#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:09:54 2024

@author: tracylin
"""

from __future__ import print_function
import requests
import json
import cv2
import trimesh
addr = 'http://localhost:5000'
test_url = addr + '/predict'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('/home/users/tracylin/Documents/AWS-3D-Mesh/experiments/ivy.png')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response

mesh = response.text
f = open('workfile.obj', 'w', encoding="utf-8")
f.write(mesh)
f.close()


mesh = trimesh.load('workfile.obj', force='mesh')
mesh.show()
#vertices = np.array(mesh.vertices)
#faces = mesh.faces