# YOLOv5 - TensorFlow Lite Object Detection iOS Example Application

[한국어 README](README_korean.md) | [ქართული README](README_georgian.md)

**iOS Versions Supported:** iOS 12.0 and above.
**Xcode Version Required:** a recent Xcode with Swift 5 support.

## Overview

This is a camera app that continuously detects objects (bounding boxes, classes and confidence scores) in the frames seen by your device's back camera, using a [YOLOv5](https://github.com/ultralytics/yolov5) model converted to TensorFlow Lite. It runs on the GPU through the TensorFlow Lite Metal delegate and falls back to the CPU if the delegate cannot be created.

## Prerequisites

* Xcode, and the command-line tools: `xcode-select --install`. If this is a new install, run Xcode once to accept the license.

* [CocoaPods](https://cocoapods.org): `sudo gem install cocoapods`. You don't need to build TensorFlow yourself; CocoaPods downloads the TensorFlow Lite library. The first `pod install` clones the TensorFlow repository to fetch the Swift sources, so it can take a while.

* An Apple Developer account, to run on a real device.

* A real iOS device. The app needs a camera. On the Simulator it builds and launches, but shows a "Configuration Failed" alert because there is no camera.

## Building the iOS Demo App

1. Install the pods to generate the workspace:

   ```sh
   cd yolov5-ios-tensorflow-lite/
   pod install
   ```

   If you have installed the pods before and that doesn't work, try `pod update`.

2. Set up your signing identity. Copy the example file and fill in your team ID and a unique bundle identifier:

   ```sh
   cp Config/Local.xcconfig.example Config/Local.xcconfig
   ```

   `Config/Local.xcconfig` is git-ignored, so your personal signing settings are never committed.

3. Open **ObjectDetection.xcworkspace** (not the `.xcodeproj`) in Xcode.

4. Build and run on your device. Grant the camera permission, then point the camera at objects.

## Models

Two COCO-trained YOLOv5 models are bundled in [ObjectDetection/Model](ObjectDetection/Model):

| File | Notes |
| --- | --- |
| `yolov5s-fp16.tflite` | Small model. More accurate, slower. **Used by default.** |
| `yolov5n-fp16.tflite` | Nano model. Less accurate, faster. |

To switch, change `Yolov5.modelInfo` in [ModelDataHandler.swift](ObjectDetection/ModelDataHandler/ModelDataHandler.swift) to `Yolov5.nanoModelInfo`.

### Model input and output

The app reads the model's tensor shapes at load time and refuses to load a model that does not match:

* **Input:** `[1, height, width, 3]`, RGB, float, scaled to `[0, 1]`. The bundled models use `640 x 640`. Camera frames are resized to fit while keeping their aspect ratio and padded with grey (a "letterbox"), and boxes are mapped back to the original frame.
* **Output:** `[1, rows, 5 + classCount]`. Each row is `x, y, w, h, objectness, class scores...`, with the box normalised to the input size. The bundled models produce `[1, 25200, 85]` (80 COCO classes).

### Using your own model

1. Train a YOLOv5 model and export it to TensorFlow Lite with `export.py` from the [YOLOv5 repository](https://github.com/ultralytics/yolov5):

   ```sh
   python export.py --weights your_model.pt --include tflite
   ```

2. Add the `.tflite` file to `ObjectDetection/Model` and add it to the Xcode target.
3. Replace `classes.txt` with your class names, one per line, in the same order as training. There must be at least as many labels as the model has classes.
4. Point `Yolov5.modelInfo` at your model.

### Tuning detections

The thresholds are in [PrePostProcessor.swift](ObjectDetection/Utils/PrePostProcessor.swift):

* `confidenceThreshold` (default `0.25`): minimum `objectness * class score` for a detection.
* `iouThreshold` (default `0.45`): overlap above which a lower-scoring box of the same class is suppressed.
* `nmsLimit` (default `100`): maximum detections per frame.

## iOS App Details

The app is written entirely in Swift and uses the TensorFlow Lite [Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) (2.17, with the Metal delegate).

* Inference runs on its own serial queue. Frames that arrive while one is still being processed are dropped, and inference is throttled to at most one every 200 ms.
* If the GPU delegate is active, the thread-count stepper in the bottom sheet has no effect; it only applies to the CPU fallback.

## Credits and License

The app is built from the object detection sample in the [TensorFlow examples repository](https://github.com/tensorflow/examples), which is licensed under the Apache License 2.0. This project is distributed under the same license; see [LICENSE](LICENSE). The YOLOv5 models come from [Ultralytics](https://github.com/ultralytics/yolov5).
