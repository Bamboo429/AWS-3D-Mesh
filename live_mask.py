#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:46:23 2023

@author: tracylin
"""

from PyQt5 import QtWidgets, uic
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QDialog, QDialogButtonBox, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from Brain_system import Brain_System
from Brain_1020 import Brain_1020
import trimesh
from demo_utils import MeshSaver

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

class MainWindow(QtWidgets.QMainWindow):
    
    brain_system = pyqtSignal(int)
    headmesh = pyqtSignal(Brain_System)
    save_filename = pyqtSignal(str)
    
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('./UI/live_mask_ui.ui', self)
        self.show()
        
        self.disply_width = 640
        self.display_height = 480
        self.btn_capture.setEnabled(False)
        self.btn_recapture.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.capture = False
        
        # button click
        self.btn_start.clicked.connect(self.videoCapture)
        self.btn_capture.clicked.connect(self.imageCapture)
        self.btn_end.clicked.connect(self.closeEvent)   
        self.btn_recapture.clicked.connect(self.reCapture)
        self.btn_save.clicked.connect(self.meshSave)
        
        # action click
        self.system_select.currentIndexChanged.connect(self.index_changed)
        
        self.head_thread = None
        self.flag_headgen = False
        
    def videoCapture(self):
        
        self.btn_capture.setEnabled(True)
        self.btn_start.setEnabled(False)
        # create the video capture thread
        self.video_thread = VideoThread()
        # connect its signal to the update_image slot
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.video_thread.start()
        
      
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.img_label.setPixmap(qt_img)
    
        if self.capture:
            cv2.imshow('head', cv_img)
            self.capture = False
            self.head_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def imageCapture(self):
        """ image for generating 3D head mesh"""
        self.capture = True
        
        dialog = ConfimDialog()
        if dialog.exec():
            self.btn_capture.setEnabled(False)
            self.btn_recapture.setEnabled(True)
            self.btn_save.setEnabled(True)
            
            cv2.destroyWindow("head")     
            
            msgBox = QMessageBox()
            msgBox.setText("Generating 3D head model")
            msgBox.exec()

            self.head_generate()
            
            #print("Success!")
        else:
            self.btn_capture.setEnabled(True)
            cv2.destroyWindow("head")  
            #print("Cancel!")
            
    @pyqtSlot(Brain_System)  
    def headmesh_capture(self, brain_system):
        
        self.bs = brain_system
        #print(brain_system.image_height)
     
    @pyqtSlot(bool)
    def head_gen_status(self, finished):
        self.flag_headgen = finished
        msgBox = QMessageBox()
        msgBox.setText("Generated 3D Head Model Successfully!")
        msgBox.exec()
        
        
        if self.flag_headgen:

            # create the head thread
            self.head_thread = HeadThread(self.head_image, self.bs)
            
            # connect its signal to the update_image slot
            self.head_thread.change_pixmap_signal.connect(self.update_image)
            # connect system selection signal to head thread
            self.brain_system.connect(self.head_thread.system_select)
            
            #self.headmesh.connect(self.head_thread.headmesh_capture)
            #self.headmesh.emit(self.bs)
            
            # start the thread
            self.video_thread.stop()
            self.head_thread.start()
            #self.headmesh.emit(self.bs)
            
        
    def head_generate(self):
          
        start_time = time.time()
        self.head_gen = HeadGenerate(self.head_image)
        end_time = time.time()
        print(f'running time : {end_time-start_time}')
        
        self.head_gen.headmesh.connect(self.headmesh_capture)
        self.head_gen.finished.connect(self.head_gen_status)
        self.head_gen.start()
        
    def meshSave(self):
        
        file_name, done = QtWidgets.QInputDialog.getText(
             self, 'Input Dialog', 'Enter your file name:')
        
        if done:
            #print(file_name)
            self.save_thread = SaveThread(self.bs, file_name)
            self.save_thread.start()
        else:
            pass
            #print('cancel', file_name)      
            
        
    def index_changed(self, idx):
        
        self.brain_system.emit(idx)
        #print(idx)

    def reCapture(self):
           
        if self.head_thread and self.head_thread.isRunning():
            self.head_thread.stop()
            
        self.videoCapture()
        self.btn_capture.setEnabled(True)
        
    def closeEvent(self, event):
        
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        
        if self.head_thread and self.head_thread.isRunning():
            self.head_thread.stop()
        #event.accept()
        self.close()
        


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    #test_signal =  pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        
    def run(self):
        # capture from web cam
        #self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('./experiments/ivy_video.mov')
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as holistic:                
            while self._run_flag:
                ret, cv_img = self.cap.read()
                if ret:
                    results = holistic.process(cv_img)
                    bs = Brain_System(cv_img)
                    if(results.face_landmarks is not None):
                        lpa, rpa = bs.predict_lpa_rpa(results.face_landmarks)
                        cv_img = bs.paint_ratio_2d_points(cv_img, np.vstack((lpa,rpa)),  (80, 175, 76))
                    
                    #print(results.face_landmarks)
                    self.change_pixmap_signal.emit(cv2.flip(cv_img, 1))
                    #self.test_signal.emit(np.asarray([0,0,0,0]))
            
        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.cap.release()
        self.wait()
        
class HeadGenerate(QThread):
    
    headmesh = pyqtSignal(Brain_System)
    finished = pyqtSignal(bool)
    
    def __init__(self, image):
        
        super().__init__()
        self._run_flag = True
        self.image = image
        
    def run(self):
        
        #print('Gernerating 3d Head Mesh......')
        self.bs = Brain_System(self.image)
        frame_height, frame_width, _ = self.image.shape
        self.frame_normalize = [frame_width, frame_height, frame_width]     
        self.bs.create_3d_head(self.image)       
        
        self.headmesh.emit(self.bs)
        self.finished.emit(True)
        
        QThread.currentThread().quit()
        
        
    def stop(self):
        
        self._run_flag = False
        self.wait()
        
class HeadThread(QThread):
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
     
    
    def __init__(self, image, bs):
        super().__init__()
        self._run_flag = True
        self.select_idx = 0
        self.bs = bs
        
        frame_height, frame_width, _ = image.shape
        self.frame_normalize = [frame_width, frame_height, frame_width]
        self.brain_montage(image)
        
    @pyqtSlot(int)  
    def system_select(self, idx):   
        self.select_idx = idx
    
    def brain_montage_select(self):
        
        if self.select_idx == 0:
            brain_points = self.brain_points_1020
        elif self.select_idx == 1:
            brain_points = self.brain_points_1010
        else:
            brain_points = self.brain_points_1020
            
        self.brain_points_front = brain_points['brain_points_front'] * self.frame_normalize
        self.brain_points_back = brain_points['brain_points_back'] * self.frame_normalize
        
    def brain_montage(self, image):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as holistic:
            results = holistic.process(image)
        
        
        
        if(results.face_landmarks is not None):
            lpa, rpa = self.bs.predict_lpa_rpa(results.face_landmarks)
            lpa = lpa[0]
            rpa = rpa[0]
            nz = self.bs.media_landmarks[168]
           
        #print(lpa, rpa, nz)
        #cv2.imshow('image', image)
        
        # mapping 3d head mesh to live image
        affined_head_ratio = self.bs.affine_3d_head()
        
        # cal brain 1020 points on the 3d head mesh
        trimesh_obj = trimesh.Trimesh(vertices = affined_head_ratio, faces = np.asarray(self.bs.faces, dtype=int))
        self.brain_1020 = Brain_1020(trimesh_obj)
        dad_index = self.brain_1020.dad_index
        
        
        self.brain_points_1010 = self.brain_1020.montage_brain_1010()
        self.brain_points_1020 = self.brain_1020.montage_brain_1020()
        #self.brain_montage_select()
        
        #brain_points = brain_points_1020
        #self.brain_points_front = brain_points['brain_points_front'] * self.frame_normalize
        #self.brain_points_back = brain_points['brain_points_back'] * self.frame_normalize
        
        # index for tranformation
        
        #media_index = [33, 263, 1, 61, 291]
        #media_index2 = [243, 463, 61, 291]
        
        self.dad_index = [2430, 3560, 1163, 3564, 3399, 819, 1515]
        self.media_index = [130, 6, 359, 1, 152, 10, 164]
        # get the landmarkd for transformation
        self.rigid_source_points = np.vstack((self.bs.media_landmarks_pixel[self.media_index]))
        self.transformed_head = affined_head_ratio*self.frame_normalize
        
    
    def run(self):
        # capture from web cam
        self.cap = cv2.VideoCapture('/home/users/tracylin/Documents/AWS-3D-Mesh/experiments/ivy_video.mov')
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #test = np.asarray([0,0,0,0])
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as holistic:                
            while self._run_flag:
                ret, cv_img = self.cap.read()
                if ret:
                    results = holistic.process(cv_img)
                    
                    if(results.face_landmarks is not None):
                        
                        
                        self.brain_montage_select()
                        lpa, rpa = self.bs.predict_lpa_rpa(results.face_landmarks)
                        
                        # rigid transform mapping
                        rigid_dst_points = np.vstack((self.bs.media_landmarks_pixel[self.media_index]))   
                        homography, transformed_head, rigid_source_points = self.bs.rigid_3d_head(self.rigid_source_points, rigid_dst_points, self.transformed_head)  
                        
                        #mapping brain 1020 points to live image
                        brain_points_front = self.bs.transform_brain_points(self.brain_points_front, homography)
                        brain_points_back = self.bs.transform_brain_points(self.brain_points_back, homography)
                       # print(brain_points_front)
                        
                        # draw mesh on the frame
                        cv_img = self.bs.paint_2d_mesh(cv_img, transformed_head)                    
                        cv_img = self.bs.paint_2d_points(cv_img, brain_points_front, (255, 144, 30))
                        #cv_img = self.bs.paint_2d_points(cv_img, brain_points_back, (235, 206, 135))
                        
                        #cv_img = self.bs.paint_ratio_2d_points(cv_img, np.vstack((lpa,rpa)),  (80, 175, 76))
                    
                    #print(results.face_ landmarks)
                    
                    self.change_pixmap_signal.emit(cv2.flip(cv_img, 1))
            
        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.cap.release()
        self.wait()
        
    
    def show_info_messagebox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
      
        # setting message for Message Box
        msg.setText("Creating 3D Head Model")
          
        # setting Message box window title
        msg.setWindowTitle("Information MessageBox")
          
        # declaring buttons on Message Box
        #msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
          
        # start the app
        retval = msg.exec_()
    
class SaveThread(QThread):
    
    def __init__(self, bs, filename):
        super().__init__()
        self._run_flag = True
        self.output_path = f'./output/{filename}.obj'
        self.bs = bs
     
    @pyqtSlot(str)
    def get_name(self, filename):
        self.output_path = f'{filename}.obj'#gui_save.obj'
     
    #@pyqtSlot(str)
    def run(self):
    
        #output_path = 
        mesh = self.bs.affined_head, self.bs.faces
        mesh_save = MeshSaver()
        mesh_save(mesh, self.output_path)
        
        QThread.currentThread().quit()
        #print('saved')
        
    def stop(self):
        
        self._run_flag = False   
        self.wait()
        
class ConfimDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("HELLO!")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel("Using this image for gererating 3D head mesh?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        

        
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()