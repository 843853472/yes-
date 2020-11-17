# -*- coding: utf-8 -*-
from pkg_resources import resource_filename

def pose_predictor_model_location():
    return resource_filename(__name__, "models/shape_predictor_68_face_landmarks.dat")

def pose_predictor_five_point_model_location():
    return resource_filename(__name__, "models/shape_predictor_5_face_landmarks.dat")

def face_recognition_model_location():
    return resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")

def cnn_face_detector_model_location():
    return resource_filename(__name__, "models/mmod_human_face_detector.dat")

def deploy_prototxt():
    return resource_filename(__name__, "models/deploy.prototxt")

def opencv_face_detector_pbtxt():
    return resource_filename(__name__, "models/opencv_face_detector.pbtxt")

def opencv_face_detector_uint8_pb():
    return resource_filename(__name__, "models/opencv_face_detector_uint8.pb")

def res10_300x300_ssd_iter_140000_fp16_caffemodel():
    return resource_filename(__name__, "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

def shape_predictor_68_face_landmarks_dat():
    return resource_filename(__name__, "models/shape_predictor_68_face_landmarks.dat")

def simple_CNN_530_0_65_hdf5():
    return resource_filename(__name__, "models/simple_CNN.530-0.65.hdf5")