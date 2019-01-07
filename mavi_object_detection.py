from AnimalDetectionModule.model import ADModel
from AnimalDetectionModule import od_utils
from PIL import Image

md = ADModel()
md.setup()

path_to_image = "obj_det/test_images/image1.jpg" #TODO full path from root to sample image
image = Image.open(path_to_image)
image_np = od_utils.load_image_into_numpy_array(image)
res = md.run(image_np)
