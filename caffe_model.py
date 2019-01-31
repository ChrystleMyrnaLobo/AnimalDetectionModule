import numpy as np
import sys,os
import cv2
import caffe
import caffe_utils as ut

class ADModel:
    """ Animal detection caffe model inference """
    def __init__(self):
        self.od_dir = "~/MobileNet-SSD/models" #TODO full path from root
        self.model_name = "mobilenet" #TODO folder name
        self.path_to_model = os.path.join(self.od_dir, self.model_name, 'mobilenet_iter_73000.caffemodel') #TODO caffemodel name
        self.path_to_net = os.path.join(self.od_dir, self.model_name, 'deploy.prototxt') #TODO prototxt name

        # Path to category label file
        file_label_map = 'mavi_animal_label_map.pbtxt'
        self.path_to_labels = os.path.join('misc', file_label_map)
        self.checkPaths()

    def checkPaths(self):
        if not os.path.exists(self.path_to_model):
            print(self.path_to_model + " does not exist")
            exit()
        if not os.path.exists(self.path_to_net):
            print(self.path_to_net + " does not exist")
            exit()
        if not os.path.exists(self.path_to_labels):
            print(self.path_to_labels + " does not exist")
            exit()

    def setup(self):
        self.category_index = load_category_index(self.path_to_labels)
        self.net = caffe.Net(self.path_to_net, self.path_to_model, caffe.TEST)

    def str(self):
        txt = "ADModel " + self.path_to_model
        txt = "ADModel " + self.path_to_net
        txt += "\nCategory Label " + self.path_to_labels
        return txt

    def run(self, image_np):
        """ Run inference for image (in numpy array)
            Output format: 1 x 1 x number of prediction x 7
              where _ x category id x score x xmin x ymin x xmax x ymax
            Top left is 0,0
        """
        self.net.blobs['data'].data[...] = image_np
        dt = self.net.forward()
        return dt
