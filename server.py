#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:39:12 2024

@author: tracylin
"""
import io
import json
from flask import Flask, jsonify, request, send_from_directory, send_file
import numpy as np
import cv2
import torch

from predictor import FaceMeshPredictor
from demo_utils import MeshSaver

app = Flask(__name__)

# The absolute path of the directory containing mesh file for users to download
app.config["3D_MESH"] = 'output/'
output_path = 'output/example.obj'
filename = 'example.obj'

@app.route('/predict', methods=['POST'])
def predict():
    
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # DAD-3D head model
    predictor = FaceMeshPredictor.dad_3dnet()
    predictions = predictor(img)
    
    vertices = predictions['3d_vertices'].numpy()
    faces = torch.load('model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
    save_3d_mesh(vertices, faces)
    
    return send_file(output_path, as_attachment=True)
    #return send_from_directory(app.config["3D_MESH"], filename=filename, as_attachment=True)
   
                    
                    
def save_3d_mesh(vertices, faces):
    
    mesh = vertices, faces
    mesh_save = MeshSaver()
    mesh_save(mesh, output_path)
    
if __name__ == '__main__':
    app.run()
