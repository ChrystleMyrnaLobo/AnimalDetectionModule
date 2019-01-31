import numpy as np
import cv2

def preprocess(src):
    """ Preprocess image as per trained data """
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, dt):
    """ Convert output as bb category and score"""
    h = img.shape[0]
    w = img.shape[1]
    bb = dt['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    category = dt['detection_out'][0,0,:,1]
    score = dt['detection_out'][0,0,:,2]
    return (bb.astype(np.int32), score, category)

def load_category_index(path_to_labels):
    return ('background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

def load_image_into_numpy_array(path_to_image):
    origimg = cv2.imread(path_to_image)
    img = ut.preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
