# Drone_Detection
Drone Detection Based on Yolo and optimized with TKDNN and TensorRT

This Drone Detector is based on Darknet([YOLO](https://github.com/AlexeyAB/darknet/)) and under process.
This detector works as a part in a long-term drone tracking system. It works both on visual domain and thermal domain

## Demo
#### Thermal Domain
![IR1](./demo/IR1.gif)
![IR3](./demo/IR3.gif)
#### Visual Domain
![visible](./demo/visible.gif)

## Speed comparison 

Experiments on RTX 2060
Resolution|Config|Speed,fps|TkDNN TensorRT speedup(fp16),fps
---|:--:|---:|---:
416*416|yolov4|82|162
512*512|yolov4|69|134
608*608|yolov4|53|103
416*416|yolov4tiny|300|790
512*512|yolov4tiny|140|-
608*608|yolov4tiny|89|-
416*416(cpu)|yolov4tiny|4|42


The library is compiled as .so file which can be used in c++(./src/main) or python environment(./python)
The code is been optimized.

## TODO

+ C++ multithread and memory safe 
+ Intergrate tracking under C++ framework
+ Provide TKDNN optimized model and test script

