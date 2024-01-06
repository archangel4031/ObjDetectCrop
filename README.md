# Object Detection using YOLOv5

## Introduction

This is a project that uses YOLOv5 model to detect objects in an image. It also crops the detected objects into separate images. Additionally, it can fetch images from URLs for object detection.

An alternate implementation of this project is available under [Gradio Folder](https://github.com/archangel4031/ObjDetectCrop/tree/main/Gradio). The user interface is built using [Gradio](https://gradio.app/).

## Usage

Before running, make sure you have setup your Anaconda environment properly.
Run the following command to start the demo:

```bash
python  main.py
```

This will perform object detection on a pre-defined image. To use your own image run

```bash
python .\main.py -i "C:/path/to/your/image.jpg" -c 0.25
```

This will perform object detection on the image and show bounding boxes around detected objects that have confidence score greater than 0.25

A detailed list about command line flags is as below
|Flag|Long Command|Usage|
|--|--|--|
|-c|--confidence|Confidence threshold value between 0 and 1. Defaults to 0.25|
|-i|--image|Path to the image file or URL to download the image. Defaults to "examples\example (1).jpg"|
|-p|--predict|Run prediction function instead of test (display summary only)|

## Limitations

It can only process a single image at a time.

## Interface Images (Gradio Version)

![image.png](<https://github.com/archangel4031/ObjDetectCrop/blob/main/gitImages/interface%20(1).png?raw=true>)

![image.png](<https://github.com/archangel4031/ObjDetectCrop/blob/main/gitImages/interface%20(2).png?raw=true>)

![image.png](<https://github.com/archangel4031/ObjDetectCrop/blob/main/gitImages/interface%20(3).png?raw=true>)
