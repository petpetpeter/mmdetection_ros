# mmdetection_ros
A set of python scripts implementing MMDetection for object detection with a depth camera.


![miniyellow](https://user-images.githubusercontent.com/55285546/137327224-e73b6477-19bd-483c-a2c5-71dba517235b.gif)



## Preparation
1. Clone this repo to your catkin workspace
```
cd /catkin_ws/src
git clone https://github.com/petpetpeter/mmdetection_ros.git
```
2. Build your workspace
```
cd /catkin_ws/
catkin build
source ~/.bashrc
```
3. Install Python Dependencies (Pytorch, MMDetection)
> follow: https://mmdetection.readthedocs.io/en/latest/get_started.html#installation (recommend using conda)

ez installation
```
conda create -n ezmmd python=3.8
conda activate ezmmd
install pytorch (follow here: https://pytorch.org/get-started/locally/)
pip install openmim
mim install mmdet
```

4. Download pretrain weight to /scripts/checkpoints
```
cd script/checkpoints
wget "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
```

5. Launch depth camera simulation 
```
roslaunch mmd_ros coco_camera.launch
```

6. Run detection script
```
conda activate ezmmd
cd /script
python rosimage_detector.py
```



![ezgif com-gif-maker](https://user-images.githubusercontent.com/55285546/137414960-87923703-37f9-4523-9f6d-6454ce6bbe73.gif)


> TensorRT 8.0.3.4: Follow installation guide here https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
1. Convert pytorch weight file to onnx file
```
roscd yoloros/scripts/yolor
python convert2Onnx.py --weight weights/yolor_p6.pt --out onnx/yolor_p6.onnx 
```
2. Convert onnx to tensorRT Engine
```
/usr/src/tensorrt/bin/trtexec --explicitBatch \
                                --onnx=transformer_encoder.onnx \
                                --saveEngine=transformer_encoder.trt \
                                --minShapes=input:1x3x32x128 \
                                --optShapes=input:32x3x32x512 \
                                --maxShapes=input:32x3x32x768 \
                                --verbose \
                                --fp16
```
3. Launch ROS Yolor TensorRT example
```
roslaunch yoloros trt_camera.launch
```



## Acknowledgement
- Yolor: https://github.com/WongKinYiu/yolor
- Yolor-TensorRT: https://github.com/NNDam/yolor

