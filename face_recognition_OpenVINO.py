###
 # @author	Amruth Karun
 # @file        face_recognition.py
 # @brief       This python code is used for face detection and recognition using OpenVINO and OpenCV
 #              ALL RIGHTS RESERVED.
 ###

#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import cv2
import numpy as np
import time
import math
import logging as log
import json
from collections import namedtuple
from openvino.inference_engine import IENetwork, IEPlugin, IENetLayer
import face_recognition

    
input_vid = "./facerecognition_humanDetection.avi"
# 

# path of the directory where OpenVINO is installed
OPENVINO_HOME = "/opt/intel/computer_vision_sdk_2018.5.455"
# path to cpu extension library
cpu_extension = OPENVINO_HOME +"/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so"
# set the probability thres
prob_threshold = 0.5
#model_xml = OPENVINO_HOME +"/deployment_tools/intel_models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
# path to model xml
model_xml = OPENVINO_HOME + "/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
# path to intermediate .bin file of person detection model
model_bin = os.path.splitext(model_xml)[0] + ".bin"
# plugin initialization for specified device and load extensions library
plugin = IEPlugin(device="CPU", plugin_dirs=None)
plugin.add_cpu_extension(cpu_extension)
# read IR of person Detection model
net_person_detection = IENetwork(model=model_xml, weights=model_bin)

# create input and output blob
input_person_detection_blob = next(iter(net_person_detection.inputs))
out_person_blob = next(iter(net_person_detection.outputs))

# load the person detection model
exec_net_person_detection = plugin.load(network=net_person_detection, num_requests=2)

# read and pre-process input frame
n, c, h, w = net_person_detection.inputs[input_person_detection_blob].shape

del net_person_detection
# start capturing the input video
cap = cv2.VideoCapture(0)

cur_request_id = 0
next_request_id = 0

# Load a sample picture and learn how to recognize it.
amruth_image = face_recognition.load_image_file("amruth.jpg")
amruth_face_encoding = face_recognition.face_encodings(amruth_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    amruth_face_encoding

]
known_face_names = [
    "Amruth"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

#cap.set(cv2.CAP_PROP_FPS, 0.05)
i=0
while cap.isOpened():  
    ret, frame = cap.read()
     
    if not ret:
       break
    
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    # resize the frame to the size of the input blob
    in_frame_person_detection = cv2.resize(frame, (w, h))
    in_frame_person_detection = in_frame_person_detection.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame_person_detection = in_frame_person_detection.reshape((n, c, h, w))

    # Start execution
    exec_net_person_detection.start_async(request_id=next_request_id, inputs={input_person_detection_blob: in_frame_person_detection})
    if exec_net_person_detection.requests[cur_request_id].wait(-1) == 0:

        # parse person detection results of the current request
        res_person_detection = exec_net_person_detection.requests[cur_request_id].outputs[out_person_blob]
        xTopLeft = []
        yTopLeft = []
        xBotRight = []
        yBotRight = []
        out_prob = []
        face_locations = []
        rgb_frame = frame[:, :, ::-1]
        for person in res_person_detection[0][0]:
            # detect person when probability is more than specified threshold of 0.5
            if person[2] > prob_threshold:
                
                # bounding box co-ordinate output of person detection
                xmin = int(person[3] * initial_w)
                ymin = int(person[4] * initial_h)
                xmax = int(person[5] * initial_w)
                ymax = int(person[6] * initial_h)
                face_locations.append((ymin,xmax,ymax,xmin))
                #print(face_locations)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_names = []
                
                for face_encoding in face_encodings:
                  # See if the face is a match for the known face(s)
                  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                  name = "Unknown"

                  # If a match was found in known_face_encodings, just use the first one.
                  if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                    
                  face_names.append(name)

                #class_id = int(person[1])
                #cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(167,255,0),1)
                #i+=1
                
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # display the detected results
    cv2.imshow('Detection Results',frame) 
    key = cv2.waitKey(1)

    if key == 27:
      break
        
  
del exec_net_person_detection
del plugin

