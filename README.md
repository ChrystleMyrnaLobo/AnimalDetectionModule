# Animal detection module
Object detection package with caffe
SSD MobileNet tensorflow models were not suppoted by Intel Movidius NCS SDK2. Hence shifting to caffe.

### Initial setup
- Install [ssd caffe]. Ensure paths are set correctly.
- Copy the pretrained model folder (having`*.prototxt` and `*.caffemodel`) into `MobileNet-SSD/models/`
- Set path in file `caffe_model.py` and `mavi_object_detection.py`. (Search for `#TODO`)
- The conda environment (saved at `misc\conda_env_ssd.yml`) can be loaded as `conda env create -f conda_env_ssd.yml`.

### To run
Run `mavi_object_detection.py` from one level up. It runs inference for file at `obj_det/test_images/image1.jpg`

### Model and dataset
Model is fine tuned on Animal dataset (dog and cow) obtained from [website].

[website]: http://www.cse.iitd.ac.in/mavi/datasets.html
[ssd caffe]: https://github.com/chuanqi305/MobileNet-SSD
