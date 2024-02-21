#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:50:04 2023

@author: tracylin
"""
import scipy.io
from mediapipe.framework.formats import landmark_pb2
from dad_utils import affinemap, transform, landmark2numpy, estimate_rigid_transformation_matrix, transform_points
import numpy as np
from DAD_3DHeads import DAD_3DHeads
import json
import cv2

class Brain_System(DAD_3DHeads):
    def __init__(self, image):
        #super().__init__(image)
        self.read_model()
        self.image_height, self.image_width, _ = image.shape
        
    def create_3d_head(self, image):
        super().__init__(image)
        
    def read_model(self):
        model_folder = './brain_model'
        
        lpa_mat = scipy.io.loadmat(f'{model_folder}/x_lpa_first_fold_of_5.mat')
        self.x_lpa = lpa_mat['x_lpa']
        
        rpa_mat = scipy.io.loadmat(f'{model_folder}/x_rpa_first_fold_of_5.mat')
        self.x_rpa = rpa_mat['x_rpa']

        iz_mat = scipy.io.loadmat(f'{model_folder}/x_iz_first_fold_of_5.mat')
        self.x_iz = iz_mat['x_iz']
        
        cz_mat = scipy.io.loadmat(f'{model_folder}/x_cz_first_fold_of_5.mat')
        self.x_cz = cz_mat['x_cz']
        
        f = open('brain_model/atlas_19-5_10-5.json')
        self.atlas10_5 = json.load(f)
       
    def get_media_predictor(self, media_head):
        
        """
        landmark=[
            pts[33],pts[133],pts[168],pts[362],pts[263],pts[4],pts[61],pts[291],pts[10],pts[332],pts[389],pts[323],pts[397]\
        pts[378],pts[152],pts[149],pts[172],pts[93],pts[162],pts[103]])
        """
        self.media_landmarks = landmark2numpy(media_head)
        self.media_landmarks_pixel = self.media_landmarks *[self.image_height, self.image_width, self.image_height]
        
        media_index = [33, 133, 168, 362, 263, 4, 61, 291, 10, 332, 389, 323, 397, 378, 152, 149, 172, 93, 162, 103]
        predictor_landmark = self.media_landmarks[media_index]
        self.predictor_landmark = np.reshape(predictor_landmark, (1, -1))
    
    def my_reg1020(self, predictor, x_pa):
        
        bsubmat = np.eye(3);
        amat = np.zeros((3,x_pa.shape[0]-3));
        ptnum = 1;
        for i in range(ptnum):
            amat[((i+1)*3-3):((i+1)*3),:]=np.kron(bsubmat,predictor[i,:])
        amat=np.hstack((amat,np.tile(bsubmat,(ptnum,1))))
        predicted_p = np.dot(amat, x_pa)
        return predicted_p
        
    def predict_lpa_rpa(self, media_head):
        
        self.get_media_predictor(media_head)
        
        self.predicted_lpa = np.transpose(self.my_reg1020(self.predictor_landmark, self.x_lpa))
        self.predicted_rpa = np.transpose(self.my_reg1020(self.predictor_landmark, self.x_rpa))
   
        return self.predicted_lpa, self.predicted_rpa
    
    def define_affine_points(self):
        
        media_index = [130, 6, 359, 1, 152]
        #media_index = [33, 133, 362, 263, 94, 78, 308]
        
        
        dad_index_lparpa = [2430, 3560, 1163, 3564, 3399, 1515, 819]
        dad_index_lparpa = [2430, 3560, 1163, 3564, 3399, 288, 1076]
        dad_index_lparpa = [2430, 3560, 1163, 3564, 3399, 272, 1051]
        #dad_index_lparpa = [2431, 3590, 3813, 1164, 3508, 2833, 2811, 1515, 819]
        
        
        dad_affine_points = self.vertices[dad_index_lparpa, :]
        
        media_affine_points = np.vstack((self.media_landmarks[media_index], self.predicted_lpa, self.predicted_rpa))
        #print(media_affine_points.shape)
        
        #dad_affine_points = self.vertices[dad_index_lparpa, :]   
        #media_affine_points = self.media_landmarks[media_index]
        
        return media_affine_points, dad_affine_points
    
    def affine_3d_head(self):
             
        #w, h, c = self.shape
                
        #media_landmarks = landmark2numpy(media_head)
        #media_index = [130, 6, 359, 1, 152]
        #dad_index = [2430, 3560, 1163, 3564, 3399] #[37,28,46,31,9]
        
        #dad_affine_points = self.vertices[dad_index, :]    
        #media_affine_points = media_landmarks[media_index, :]
        media_affine_points, dad_affine_points = self.define_affine_points()
        #media_affine_points = media_affine_points*[self.image_height, self.image_width, self.image_height]
        
        #media_affine_points = [[1,1,1],[20,20,20],[3,4,5],[5,8,10],[150,100,30]]
        A, b = affinemap(dad_affine_points, media_affine_points)       
        dad_head = transform(A,b, self.vertices)
        
        self.affined_head = dad_head
        #print(dad_affine_points)
        
        return dad_head
    
    def get_brain_system_anchor(self, media_head):
        
        self.get_media_predictor(media_head)
        
        nz = self.media_landmarks[168]
        iz = np.transpose(self.my_reg1020(self.predictor_landmark, self.x_iz))[0]
        cz = np.transpose(self.my_reg1020(self.predictor_landmark, self.x_cz))[0]
        lpa = self.predicted_lpa[0]
        rpa = self.predicted_rpa[0]
        
        #print(nz, iz, cz, lpa, rpa)
        return np.vstack((nz, lpa, rpa, iz, cz))
    
    def affine_brain_system(self, media_head):
        
        predicted_anchor = self.get_brain_system_anchor(media_head)
        brain_system_anchor = np.array([self.atlas10_5['nz'], self.atlas10_5['lpa'], self.atlas10_5['rpa'], 
                                        self.atlas10_5['iz'], self.atlas10_5['cz']]) 
                       
        #print(brain_system_anchor.shape, predicted_anchor.shape)
        A, b = affinemap(brain_system_anchor, predicted_anchor)   
        
        brain_system = {}
        for k in self.atlas10_5.keys():
            brain_system[k] = transform(A, b, self.atlas10_5[k])
        
        return brain_system
        
        
    def paint_brain_system(self, frame, media_head):
        
        no_print_keys = ['paal', 'papl', 'paar', 'papr', 'apl', 'apr', 'cz']
        print_keys = ['aal', 'aar', 'sm', 'cm', 'cal_1', 'car_1', 'cal_2', 'car_2'
                      'cal_3', 'car_3', 'cal_4', 'car_4', 'cal_5', 'car_5',
                      'cal_6', 'car_6', 'cal_7', 'car_7']
        p=0
        w, h, c = frame.shape
        brain_system = self.affine_brain_system(media_head)
        for k in brain_system.keys():
            if k in print_keys:
                brain_points = brain_system[k]
                for point in brain_points:
                    p+=1
                    point_2d = (np.ceil(point[0]*h).astype(int), np.ceil(point[1]*w).astype(int))
                    #print(p, point_2d)
                    color = (255, 255, 0)
                    
                    frame = cv2.circle(frame, point_2d, radius=5, color=color, thickness=5)
        
        #print (p)
        return frame
    
    def rigid_3d_head(self, pre_feature_points, now_feature_points, points):
        
    
        H = estimate_rigid_transformation_matrix(pre_feature_points, now_feature_points)
        transformed_points = transform_points(points, H)       
   
        return H, transformed_points, now_feature_points
    
    def ransac_rigid_3d(self, pre_feature_points, now_feature_points, points, sample_size=10, iteration=50):
        
        best_error = np.inf
        
        n_samples, _ = pre_feature_points.shape
        sample = np.random.choice(n_samples, sample_size)
        
        source_point = pre_feature_points[sample]
        dst_point = now_feature_points[sample]
        
        for _ in range(iteration):
           H = estimate_rigid_transformation_matrix(pre_feature_points, now_feature_points)
           transformed_points = transform_points(source_point, H)  
           
           error = np.sum(np.abs(transformed_points - dst_point))
           #print(error)
           if best_error > error:
               transformed = transformed_points
               best_error = error
               best_H = H
               best_point = dst_point
        
            
        transformed_points = transform_points(points, H)  
        return  H, transformed_points, now_feature_points, best_point
            
    def transform_brain_points(self, points, H):
        
        transformed_points = transform_points(points, H)  
        
        return transformed_points

f = open('brain_model/atlas_19-5_10-5.json')
atlas10_5 = json.load(f)      
keys = list(atlas10_5.keys())
brain_system = {}
for k in atlas10_5.keys():
    brain_system[k] = 0
    #print (k)
    
        
        
        
        
        
        


        
        
        
        
