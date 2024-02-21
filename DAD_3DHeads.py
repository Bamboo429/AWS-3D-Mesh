#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:08:49 2023

@author: chuhsuanlin
"""

#from predictor import FaceMeshPredictor
from utils import get_relative_path
import numpy as np
import cv2
import os
#import torch
#from demo_utils import MeshSaver
from dad_utils import affinemap, transform, landmark2numpy
#from model_training.head_mesh import HeadMesh

import trimesh
import requests
# for test
#from pytorch_toolbelt.utils import read_rgb_image
import matplotlib.pyplot as plt

# region visualization
POINT_COLOR = (255, 0, 0)
EDGE_COLOR = (39, 48, 218)
OPACITY = .6
KEYPOINTS_INDICES_DIR = "model_training/model/static/face_keypoints"
FLAME_IDICES_DIR = "model_training/model/static/flame_indices/"


class DAD_3DHeads():
    """
    
    """
    
    def __init__(self, image):
        self.image = image
        self.image_width, self.image_height, self.image_channel = image.shape
        self.edges = np.load(get_relative_path(os.path.join(FLAME_IDICES_DIR, f"head_edges.npy"), __file__))
        self.face_edges = np.load(get_relative_path(os.path.join(FLAME_IDICES_DIR, f"face_edges.npy"), __file__))

        #self.head_mesh = HeadMesh()
        self.get_3d_head()
        #self.save_3d_mesh('output/capture_3d.obj')
        #self. predictor = FaceMeshPredictor.dad_3dnet()
    def get_3d_head(self):

        
        addr = 'http://3.145.64.155:5000'
        #addr = 'http://127.31.10.85:5000'
        test_url = addr + '/predict'
        
        # prepare headers for http request
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}

        img = self.image
        # encode image as jpeg
        _, img_encoded = cv2.imencode('.jpg', img)
        
        
        
        # send http request with image and receive response
        response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
        print("mesh")
        # decode response
        mesh = response.text
        
        f = open('workfile.obj', 'w', encoding="utf-8")
        f.write(mesh)
        f.close()
        
        mesh = trimesh.load('workfile.obj', force='mesh')
        self.vertices = np.array(mesh.vertices)
        self.faces = np.array(mesh.faces)
        
        
        '''
        image = self.image
        predictor = FaceMeshPredictor.dad_3dnet()
        self.predictions = predictor(image)
        
        self.vertices = self.predictions['3d_vertices'].numpy()
        self.faces = torch.load('model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        
        self.projected_vertices = self.predictions['projected_vertices'].squeeze().numpy().astype(int)
        #print(projected_vertices)
        
        params_3dmm = self.predictions['3dmm_params']
        #print(params_3dmm)
        '''
        
    '''
    def get_continue_3d(self, image):
        #predictor = FaceMeshPredictor.dad_3dnet()
        self.predictions = self.predictor(image)
        projected_vertices = self.predictions['projected_vertices'].squeeze().numpy().astype(int)
        #print(projected_vertices)
        mesh_vis = image.copy()
        
        edges = self.edges
        
        for edge in edges:
            pt1, pt2 = edge[0], edge[1]
            cv2.line(mesh_vis, projected_vertices[pt1], projected_vertices[pt2], EDGE_COLOR, 1, cv2.LINE_AA)
    
        
        #cv2.addWeighted(mesh_vis, OPACITY, output, 1 - OPACITY, 0, output)
        return mesh_vis
    '''
    
    def save_3d_mesh(self, output_path):
        
        '''
        mesh = vertices, self.faces
        mesh_save = MeshSaver()
        mesh_save(mesh, output_path)
        '''
        
        mesh = self.vertices, self.faces
        mesh_save = MeshSaver()
        mesh_save(mesh, output_path)
        
    def define_affine_points(self, media_head):
        media_index = [130, 6, 359, 1, 152]
        dad_index = [2430, 3560, 1163, 3564, 3399]
        
        media_landmarks = landmark2numpy(media_head)
        dad_affine_points = self.vertices[dad_index, :]
        media_affine_points = media_landmarks[media_index]
        
        return media_affine_points, dad_affine_points
        
    def paint_2d_mesh(self, image, dad_transformed_head):    
    
        w, h, c = image.shape
        #dad_transformed_head = self.affine_3d_head(1)
        #print(dad_transformed_head)
        #dad_transformed_head = self.vertices
        
                
        #dad_transformed_t, _ = cv2.projectPoints(dad_transformed_head, (0,0,0), (0,0,0), camera_matrix, np.zeros(4, dtype='float32'))

        #dad_transformed_head = np.reshape(dad_transformed_t, (5023,2))
        #print(dad_transformed_head.shape)
        
        projected_dad_transformed = dad_transformed_head[:,0:2]
        #projected_dad_transformed = dad_transformed_head
        #print(projected_dad_transformed.shape)
        projected_vertices = np.ceil(projected_dad_transformed).astype(int)
        #print(projected_vertices)
        
        #projected_vertices = self.predictions["projected_vertices"].squeeze().numpy().astype(int)
        mesh_vis = image.copy()
        
        # load the edge reltaionship
        edges = self.edges
   
        for edge in edges:
            
            pt1, pt2 = edge[0], edge[1]
            cv2.line(mesh_vis, projected_vertices[pt1], projected_vertices[pt2], EDGE_COLOR, 1, cv2.LINE_AA)
    
        
        #cv2.addWeighted(mesh_vis, OPACITY, output, 1 - OPACITY, 0, output)
        return mesh_vis
    
    def paint_2d_points(self, image, points_3d, color=(0,255,0)):
     
        for point in points_3d:
            point_2d = (np.ceil(point[0]).astype(int), np.ceil(point[1]).astype(int))            
            image = cv2.circle(image, point_2d, radius=2, color=color, thickness=2)
            
        return image
    
    def paint_ratio_2d_points(self, image, points_3d, color=(0,255,0)):
        
        points_3d = points_3d*[self.image_width, self.image_height, self.image_width]   
        image = self.paint_2d_points(image, points_3d)
            
        return image
        
        

    '''
    def get_3d_mesh(self):
        
        vertices = self.predictions['3d_vertices'].numpy()
        faces = torch.load('model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        
        return vertices, faces
    '''

    def affine_3d_head(self, media_head):
        
        
        #w, h, c = self.shape
                
        #media_landmarks = landmark2numpy(media_head)
        #media_index = [130, 6, 359, 1, 152]
        #dad_index = [2430, 3560, 1163, 3564, 3399] #[37,28,46,31,9]
        
        #dad_affine_points = self.vertices[dad_index, :]    
        #media_affine_points = media_landmarks[media_index, :]
        media_affine_points, dad_affine_points = self.define_affine_points(media_head)
        
        #media_affine_points = [[1,1,1],[20,20,20],[3,4,5],[5,8,10],[150,100,30]]
        A, b = affinemap(dad_affine_points, media_affine_points)       
        dad_head = transform(A,b, self.vertices)
        
        #print(dad_affine_points)
        
        return A, b, dad_head
    
    def show_3d_mesh(self):
        
        mesh = trimesh.load_mesh('output/capture_3d.obj')
        mesh.show()
        pass

        
 
        
'''
image_path = 'experiments/test1.jpeg'
image = read_rgb_image(image_path)

DAD_3D = DAD_3DHeads(image) 
DAD_3D.get_3d_head()
#plt.scatter(DAD_3D.vertices[:,0], DAD_3D.vertices[:,1], DAD_3D.vertices[:,2])     

image_mesh = DAD_3D.paint_2d_points(image, DAD_3D.vertices)  
plt.imshow(image_mesh)   

#DAD_3D.show_3d_mesh()
#DAD_3D.save_3d_mesh('test1.obj') 
'''  
      
    
