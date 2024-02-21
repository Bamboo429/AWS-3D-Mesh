#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:05:13 2023

@author: tracylin
"""
import numpy as np

def affinemap(pfrom, pto):
    
    bsubmat=np.eye(3)
    ptnum=len(pfrom)
    amat=np.zeros((ptnum*3,9))
    for i in range(ptnum):
        amat[((i+1)*3-3):((i+1)*3),:]=np.kron(bsubmat,pfrom[i,:])
    amat=np.hstack((amat,np.tile(bsubmat,(ptnum,1))))
    
    bvec=np.reshape(pto,(ptnum*3,1))
    x=np.linalg.lstsq(amat, bvec, rcond=None)[0]
    A=np.reshape(x[0:9],(3,3))
    b=x[9:12]
    return [A, b]

def transform(Amat, bvec, pts):
    newpt = np.matmul(Amat, (np.array(pts)).T)
    newpt = newpt + np.tile(bvec, (1, len(pts)))
    newpt = newpt.T
    return newpt

def landmark2numpy(landmarks):
    pts = []
    for p in landmarks.landmark:
        pts.append([p.x, p.y, p.z])
    return np.array(pts)


def nearest_node(node, vertices):
    diff = (node-vertices)*(node-vertices)
    distance = np.sum(diff, axis = 1)
    nearest_idx = np.argmin(distance)
    
    return nearest_idx

def polylinelen(nodes):
    
    length = len(nodes)
    diff = nodes[0:length-1] - nodes[1:length]
    distance = np.sqrt(np.sum(diff*diff, axis = 1))
    
    return distance
    
def polylineinterp(nodes, seg_ratios):
    
    seglens = polylinelen(nodes)
    cumlens = np.cumsum(seglens)
    total_len = np.sum(seglens)
    
    interp_nodes = np.zeros((len(seg_ratios)+1, 3))
    interp_nodes[0] = nodes[0]
    
    
    s=0
    act_len =0
    w = np.zeros(len(seglens))
    
    length = len(nodes)
    vec = (nodes[0:length-1] - nodes[1:length])
    diff = nodes[0:length-1] - nodes[1:length]
    vec = diff/np.linalg.norm(diff)
    
    flag = False
    for i, r in enumerate(seg_ratios):
       act_len = act_len + total_len*r
       
       while cumlens[s]<=act_len:
           #print(i, cumlens[s], act_len, s)
           w[s] = 1  
           s += 1
           
           if s == len(cumlens):
               s -=1
               break
  
       w[s] = (cumlens[s]-act_len)/(seglens[s])
       interp_nodes[i+1] = nodes[s]+seglens[s]*w[s]*vec[s]
       
       #print(i+1, interp_nodes[i+1], s, cumlens[s], act_len)
       s += 1

    interp_nodes[i+1] = nodes[length-1] 
    return interp_nodes
   
def angle_between_vectors(vector1, vector2):
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosine_angle = dot_product / (magnitude1 * magnitude2)

    if np.isclose(cosine_angle, -1.0):
        angle_degrees = 180.0
    else:
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def estimate_rigid_transformation_matrix(src_points, dst_points):
    assert src_points.shape == dst_points.shape
    assert src_points.shape[1] == 3  # 3D points

    # Calculate centroids of both point sets
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Center the points around the centroids
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Calculate the scaling factor
    src_norm = np.linalg.norm(src_centered)
    dst_norm = np.linalg.norm(dst_centered)
    scaling = dst_norm / src_norm

    # Compute the covariance matrix between the centered points
    covariance_matrix = np.dot(src_centered.T, dst_centered)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Ensure that the determinant of the rotation matrix is 1 (avoid reflections)
    det = np.linalg.det(np.dot(Vt.T, U.T))
    if det < 0:
        #print(det)
        Vt[2] *= -1
    
    # Compute the rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Compute the translation vector
    translation_vector = dst_centroid - scaling * np.dot(rotation_matrix, src_centroid)

    # Combine scaling, rotation, and translation into a transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = scaling * rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix #scaling * rotation_matrix,translation_vector #


def transform_points(src_points, homography_matrix):
    # Add homogeneous coordinate (1) to the 3D source points
    src_points_homogeneous = np.hstack((src_points, np.ones((len(src_points), 1))))

    # Perform the transformation using the homography matrix
    dst_points_homogeneous = np.dot(homography_matrix, src_points_homogeneous.T).T

    # Convert back to 3D coordinates (remove the homogeneous coordinate)
    dst_points = dst_points_homogeneous[:, :3] / dst_points_homogeneous[:, 3, np.newaxis]

    return dst_points

    
    
    
    