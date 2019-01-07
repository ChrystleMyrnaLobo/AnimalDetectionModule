import os
import numpy as np
from AnimalDetectionModule.tf_inference import InferenceEngine
from AnimalDetectionModule.od_utils import *

class ADModel:
    """ Animal detection model inference """
    def __init__(self):
        self.od_dir = "models/research/object_detection" #TODO full path from root to models/research/object_detection

        # Path to frozen graph
        self.model_name = "dogcow_ssd_inception" # Model fine tuned on MAVI Animal (dog, cow) Dataset
        self.path_to_frozen_graph = self.model_name + '/frozen_inference_graph.pb'
        self.path_to_frozen_graph = os.path.join(self.od_dir, self.path_to_frozen_graph)


        # Path to category label file in models/research/object_detection/data
        file_label_map = 'mavi_animal_label_map.pbtxt'
        self.path_to_labels = os.path.join('data', file_label_map)
        self.path_to_labels = os.path.join(self.od_dir, self.path_to_labels)
        self.num_classes = 2 # num of classes model is trained on

    def setup(self):
        self.category_index = load_category_index(self.path_to_labels, self.num_classes)
        self.eng = InferenceEngine(self.path_to_frozen_graph)

    def str(self):
        txt = "ADModel " + self.path_to_frozen_graph
        txt += "\nCategory Label " + self.path_to_labels
        return txt

    def run(self, image_np):
        """ Run inference for image (in numpy array)"""
        detection_dict = []
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        detection_dict = self.eng.run_inference_for_single_image(image_np)
        # The predicition gives BB in normalized coordinated
        # Convert to original image cordinates from normalized coordinates (for evaluation and vizualization)
        im_height, im_width, _ =  image_np.shape
        detection_dict['detection_boxes'] = denormalise_box(detection_dict['detection_boxes'], (im_width, im_height))
        return detection_dict
