#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:22:52 2023

@author: tracylin
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import networkx as nx
from dad_utils import affinemap, transform, polylinelen, polylineinterp, angle_between_vectors
from DAD_3DHeads import DAD_3DHeads

  
class Brain_1020():
    def __init__(self, trimesh_obj, lpa=None, rpa=None, nz=None):
        #self.mesh = trimesh.load(trimesh_obj)
        #self.vertices = self.mesh.vertices
        self.mesh = trimesh_obj
        
        if lpa is not None:
            self.lpa = lpa
            self.rpa = rpa
            self.nz = nz
    
        else:
            dad_index_lparpa = [2430, 3560, 1163, 3564, 3399, 272, 1051]
            vertices = self.mesh.vertices[dad_index_lparpa]
            self.lpa = vertices[5]
            self.rpa = vertices[6]
            self.nz = vertices[1]
            
        self.dad_index = [2430, 3560, 1163, 3564, 3399, 272, 1051]
        
    def get_mesh(self):
        return self.mesh
    
    def ray_mesh_intersection(self, mesh, ray_origin, ray_direction):
        # Load the mesh using trimesh
        # mesh = trimesh.load(mesh_path)
        
        # Calculate the intersection between the ray and the mesh
        intersections, _, _ = mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction],
            multiple_hits=False
        )
        
        if intersections is not None and len(intersections) > 0:
            # Return the closest intersection point
            closest_intersection = intersections[0]
            return closest_intersection.tolist()
        
        # Return None if there are no intersections
        return None
    
    def get_head_landmarks(self):
        
             
        pa_mid = (self.lpa+self.rpa)/2
        v_iz = pa_mid-self.nz

        self.iz = np.array(self.ray_mesh_intersection(self.mesh, pa_mid, v_iz))

        iznz_mid = (self.iz+self.nz)/2
        v_cz = np.cross(self.nz-self.lpa, self.lpa-self.rpa)
        self.cz = np.array(self.ray_mesh_intersection(self.mesh, iznz_mid, v_cz))
        
        return [self.nz, self.iz, self.cz, self.lpa, self.rpa]
    
    '''
    def montage_nziz(self):
        
        new_v, new_f = trimesh.remesh.subdivide(self.mesh.vertices, self.mesh.faces)
        new_mesh = trimesh.base.Trimesh(new_v, new_f)
        
        new_mesh = self.mesh
        # edges without duplication
        edges = new_mesh.edges_unique
        # the actual length of each unique edge
        length = new_mesh.edges_unique_length
        
        self.graph = nx.Graph()
        for edge, L in zip(edges, length):
            self.graph.add_edge(*edge, length=L)
            
        nearest_cz = nearest_node(self.cz, new_mesh.vertices)
        nearest_iz =  nearest_node(self.iz, new_mesh.vertices)
        nearest_nz = nearest_node(self.nz, new_mesh.vertices)
        
        path_nzcz = nx.shortest_path(self.graph,
                                source=nearest_nz,
                                target=nearest_cz,
                                weight='length')
        
        path_cziz = nx.shortest_path(self.graph,
                                source=nearest_cz,
                                target=nearest_iz,
                                weight='length')
        
        path = np.concatenate((path_nzcz, path_cziz))
        
        #seg_distance = polylinelen(self.mesh.vertices[path])
        
        cz  = polylineinterp(new_mesh.vertices[path], [0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        
        cc = []
        for c in cz:
            cc.append(new_mesh.vertices[nearest_node(c, new_mesh.vertices)])
        return path, cc
    '''
    def get_brain_points(self, ori_point, start_point, end_point, reference_plane, slice_ratio):
        
        
        if not np.isclose(np.sum(slice_ratio), 1):
            #print (np.sum(seg_ratios))
            print('Sum of slice_ratio mush be 1 ')
            return False
        
        
        vector1 = start_point - ori_point
        vector2 = end_point - ori_point
        angle = angle_between_vectors(vector1, vector2)
        
        points = []
        for r in np.arange(0, angle+1, 1):
            #print(r)
            angle_radians = np.radians(r)
            #print(np.rad2deg(angle_radians))
        
            rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, reference_plane, ori_point)
            rotated_vector = np.dot(rotation_matrix[:3, :3], vector1)
            
            point = self.ray_mesh_intersection(self.mesh, ori_point, rotated_vector)
            points.append(point)
            
        
       
        brain_points  = polylineinterp(np.array(points), slice_ratio)
        brain_points[len(slice_ratio)] = end_point
        
        return brain_points
        #return points
    
    def montage_brain_1020(self):
        
        self.get_head_landmarks()
        
        iznz_mid = (self.iz+self.nz)/2
        v_cz_plane = np.cross(self.iz-self.cz, self.nz-self.cz)
        
        slice_ratio_1020 = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        slice_ratio_1010 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        slice_ratio_half = [0.2, 0.4, 0.4]
        slice_ratio_cut = [0.5, 0.5]
        
        # cal point on sm
        brain_points_sm = self.get_brain_points(iznz_mid, self.nz, self.iz, v_cz_plane, slice_ratio_1020)
        self.new_cz = brain_points_sm[3]
        #self.new_cz = self.cz
        
        # cm
        pa_mid = (self.lpa+self.rpa)/2
        v_plane =  np.cross(self.lpa-self.new_cz, self.new_cz-self.rpa) 
        brain_points_cm = self.get_brain_points(pa_mid, self.lpa, self.rpa, v_plane, slice_ratio_1020)
        brain_points_cm[3] = self.new_cz
        
        
        # aal aar apl apr
        T7, T8 = brain_points_cm[1], brain_points_cm[5]
        Fpz, Oz = brain_points_sm[1], brain_points_sm[5]
        central_mid = (T7+T8)/2
        v_plane = np.cross(T7-Fpz, Oz-Fpz)
        brain_points_aal1 = self.get_brain_points(central_mid, Fpz, T7, v_plane, slice_ratio_half)
        brain_points_aal2 = self.get_brain_points(central_mid, T7, Oz, v_plane, np.flip(slice_ratio_half))
        
        v_plane = np.cross(T8-Fpz, Oz-Fpz)
        brain_points_aar1 = self.get_brain_points(central_mid, Fpz, T8, v_plane, slice_ratio_half)
        brain_points_aar2 = self.get_brain_points(central_mid, T8, Oz, v_plane, np.flip(slice_ratio_half))
        
        
        F7, F8 = brain_points_aal1[2], brain_points_aar1[2]
        Fz = brain_points_sm[2]
        central_mid = (F7+F8)/2
        v_plane = np.cross(F8-Fz, F7-Fz)
        brain_points_cutf1 = self.get_brain_points(central_mid, F7, Fz, v_plane, slice_ratio_cut)
        brain_points_cutf2 = self.get_brain_points(central_mid, Fz, F8, v_plane, slice_ratio_cut)
        
        P7, P8 = brain_points_aal2[1], brain_points_aar2[1]
        Pz = brain_points_sm[4]
        central_mid = (P7+P8)/2
        v_plane = np.cross(P8-Pz, P7-Pz)
        brain_points_cutp1 = self.get_brain_points(central_mid, P7, Pz, v_plane, slice_ratio_cut)
        brain_points_cutp2 = self.get_brain_points(central_mid, Pz, P8, v_plane, slice_ratio_cut)
            
        brain_points_front = np.vstack((brain_points_sm[1:5], brain_points_cm, 
                                        brain_points_aal1, brain_points_aar1, 
                                        brain_points_cutf1, brain_points_cutf2))
        
        #brain_points_front = np.vstack((brain_points_sm, brain_points_cm, 
        #                                ))
        
        brain_points_back = np.vstack((brain_points_aal2, brain_points_aar2, 
                                       brain_points_cutp1, brain_points_cutp2))
        
        
        brain_points = np.vstack((brain_points_sm[1:6], brain_points_cm[1:6], 
                                  brain_points_aal1[1:3], brain_points_aal2[1:3],
                                  brain_points_aar1[1:3], brain_points_aar2[1:3],
                                  brain_points_cutf1[1], brain_points_cutf2[1],
                                  brain_points_cutp1[1], brain_points_cutp2[1],
                                  self.nz, self.iz, self.lpa, self.rpa,
                                  ))
        
        test_points = np.vstack((brain_points_sm,brain_points_cm))
        #brain_points_front =[]
        #brain_points_back=[]
        #brain_points=[]
        return {'brain_points_front':brain_points_front, 
                'brain_points_back':brain_points_back,
                'brain_points': brain_points, 
                'test_points':test_points}
        '''
        return {'brain_points_front':brain_points_cm, 
                'brain_points_back':brain_points_sm,
                }
        '''
    def montage_brain_1010(self):
        
        self.get_head_landmarks()
        
        iznz_mid = (self.iz+self.nz)/2
        v_cz_plane = np.cross(self.iz-self.cz, self.nz-self.cz)
        
        slice_ratio_1020 = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        slice_ratio_1010 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        slice_ratio_half = [0.2, 0.4, 0.4]
        slice_ratio_half_1010 = [0.2, 0.2, 0.2, 0.2, 0.2]
        slice_ratio_cut = [0.5, 0.5]
        slice_ratio_cut_1010 = [0.25, 0.25, 0.25, 0.25]
        
        # cal point on sm
        brain_points_sm = self.get_brain_points(iznz_mid, self.nz, self.iz, v_cz_plane, slice_ratio_1010)
        self.new_cz = brain_points_sm[5]
        
        # cm
        pa_mid = (self.lpa+self.rpa)/2
        v_plane =  np.cross(self.lpa-self.new_cz, self.new_cz-self.rpa) 
        brain_points_cm = self.get_brain_points(pa_mid, self.lpa, self.rpa, v_plane, slice_ratio_1010)
        brain_points_cm[5] = self.new_cz
        
        # aal aar apl apr
        T7, T8 = brain_points_cm[1], brain_points_cm[9]
        Fpz, Oz = brain_points_sm[1], brain_points_sm[9]
        central_mid = (T7+T8)/2
        v_plane = np.cross(T7-Fpz, Oz-Fpz)
        brain_points_aal1 = self.get_brain_points(central_mid, Fpz, T7, v_plane, slice_ratio_half_1010)
        brain_points_aal2 = self.get_brain_points(central_mid, T7, Oz, v_plane, np.flip(slice_ratio_half_1010))
        
        v_plane = np.cross(T8-Fpz, Oz-Fpz)
        brain_points_aar1 = self.get_brain_points(central_mid, Fpz, T8, v_plane, slice_ratio_half_1010)
        brain_points_aar2 = self.get_brain_points(central_mid, T8, Oz, v_plane, np.flip(slice_ratio_half_1010))
        
        
        AF7, AF8 = brain_points_aal1[2], brain_points_aar1[2]
        AFz = brain_points_sm[2]
        central_mid = (AF7+AF8)/2
        v_plane = np.cross(AF8-AFz, AF7-AFz)
        brain_points_cutaf1 = self.get_brain_points(central_mid, AF7, AFz, v_plane, slice_ratio_cut)
        brain_points_cutaf2 = self.get_brain_points(central_mid, AFz, AF8, v_plane, slice_ratio_cut)
        
        F7, F8 = brain_points_aal1[3], brain_points_aar1[3]
        Fz = brain_points_sm[3]
        central_mid = (F7+F8)/2
        v_plane = np.cross(F8-Fz, F7-Fz)
        brain_points_cutf1 = self.get_brain_points(central_mid, F7, Fz, v_plane, slice_ratio_cut_1010)
        brain_points_cutf2 = self.get_brain_points(central_mid, Fz, F8, v_plane, slice_ratio_cut_1010)
        
        FT7, FT8 = brain_points_aal1[4], brain_points_aar1[4]
        FCz = brain_points_sm[4]
        central_mid = (FT7+FT8)/2
        v_plane = np.cross(FT8-FCz, FT7-FCz)
        brain_points_cutft1 = self.get_brain_points(central_mid, FT7, FCz, v_plane, slice_ratio_cut_1010)
        brain_points_cutft2 = self.get_brain_points(central_mid, FCz, FT8, v_plane, slice_ratio_cut_1010)
        
        TP7, TP8 = brain_points_aal2[1], brain_points_aar2[1]
        CPz = brain_points_sm[6]
        central_mid = (TP7+TP8)/2
        v_plane = np.cross(TP8-CPz, TP7-CPz)
        brain_points_cuttp1 = self.get_brain_points(central_mid, TP7, CPz, v_plane, slice_ratio_cut_1010)
        brain_points_cuttp2 = self.get_brain_points(central_mid, CPz, TP8, v_plane, slice_ratio_cut_1010)
        
        P7, P8 = brain_points_aal2[2], brain_points_aar2[2]
        Pz = brain_points_sm[7]
        central_mid = (P7+P8)/2
        v_plane = np.cross(P8-Pz, P7-Pz)
        brain_points_cutp1 = self.get_brain_points(central_mid, P7, Pz, v_plane, slice_ratio_cut_1010)
        brain_points_cutp2 = self.get_brain_points(central_mid, Pz, P8, v_plane, slice_ratio_cut_1010)
            
        PO7, PO8 = brain_points_aal2[3], brain_points_aar2[3]
        POz = brain_points_sm[8]
        central_mid = (PO7+PO8)/2
        v_plane = np.cross(PO8-POz, PO7-POz)
        brain_points_cutpo1 = self.get_brain_points(central_mid, PO7, POz, v_plane, slice_ratio_cut)
        brain_points_cutpo2 = self.get_brain_points(central_mid, POz, PO8, v_plane, slice_ratio_cut)
        
        brain_points_front = np.vstack((brain_points_sm[1:5], brain_points_cm, 
                                        brain_points_aal1, brain_points_aar1, 
                                        brain_points_cutaf1, brain_points_cutaf2,
                                        brain_points_cutf1, brain_points_cutf2,
                                        brain_points_cutft1, brain_points_cutft2))
        
        brain_points_back = np.vstack((brain_points_aal2, brain_points_aar2,
                                       brain_points_cuttp1, brain_points_cuttp2,
                                       brain_points_cutp1, brain_points_cutp2,
                                       brain_points_cutpo1, brain_points_cutpo2))
        
        brain_points = np.vstack((brain_points_sm[1:-1], brain_points_cm[1:-1], 
                                  brain_points_aal1[1:-1], brain_points_aal2[1:-1],
                                  brain_points_aar1[1:-1], brain_points_aar2[1:-1],
                                  brain_points_cutaf1[1:-1], brain_points_cutaf2[1:-1],
                                  brain_points_cutf1[1:-1], brain_points_cutf2[1:-1],
                                  brain_points_cutft1[1:-1], brain_points_cutft2[1:-1],
                                  brain_points_cuttp1[1:-1], brain_points_cuttp1[1:-1],
                                  brain_points_cutp1[1:-1], brain_points_cutp2[1:-1],
                                  brain_points_cutpo1[1:-1], brain_points_cutpo1[1:-1],
                                  self.nz, self.iz, self.lpa, self.rpa,
                                  ))
    
        return {'brain_points_front':brain_points_front, 
                'brain_points_back':brain_points_back,
                'brain_points':brain_points}
    
        '''
        return {'brain_points_front':brain_points_aar1, 
                'brain_points_back':brain_points_aar2,
                }
        '''
        
        
        
    def affine_transform(self, from_landmarks, to_landmarks, points):
        
        A, b = affinemap(from_landmarks, to_landmarks)        
        return transform(A, b, points)
        
    
    
        
        