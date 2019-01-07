# Animal detection module
Object detection package

### Initial setup
- Install [Tensorflow object detection API]. Alias `models/research/object_detection` as `obj_det`. Ensure path is set correctly.
- Set path in file `model.py` and `mavi_object_detection.py`. (Search for `#TODO`)
- Download the pre trained model into the `obj_det` directory
- Copy `misc/mavi_animal_label_map.pbtxt` into `obj_det/data` directory


### To run
Run `mavi_object_detection.py` from one level up. It runs inference for file at `obj_det/test_images/image1.jpg`

### Model and dataset
Model is fine tuned on Animal dataset (dog and cow) obtained from [website].

[website]: http://www.cse.iitd.ac.in/mavi/datasets.html
[Tensorflow object detection API]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
