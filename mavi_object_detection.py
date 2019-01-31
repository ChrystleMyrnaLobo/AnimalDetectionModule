from AnimalDetectionModule.caffe_model import ADModel
from AnimalDetectionModule import caffe_utils as ut
from PIL import Image

md = ADModel()
md.setup()

path_to_image = "test_images/image1.jpg" #TODO full path from root to sample image
image = Image.open(path_to_image)
image_np = ut.load_image_into_numpy_array(image)
res = md.run(image_np)
