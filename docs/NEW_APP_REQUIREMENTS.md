# Requirements: real-time object detection app for iPhone XS Max (iOS 18.7.8)

Requirements for a new app, written from scratch and optimized for the iPhone XS Max on iOS 18.7.8. They are based on what we learned building and tuning this project (`yolov5-ios-tensorflow-lite`).

None of the performance targets below have been measured on the device yet. Treat them as goals to confirm in the first spike (section 7), not as promises.

## 0. Utilization target

"Use 80-90% of the hardware" does not work as a literal target on this phone:

- **Thermals:** the XS Max is a 2018 device with an aging battery. Running the Neural Engine, GPU and CPU all near 90% will reach a serious thermal state within minutes, and iOS will throttle it.
- **Better goal:** keep the **Neural Engine (the main compute engine) 80-90% busy**, while the CPU and GPU stay mostly idle. That gives the highest sustainable throughput.
- **How to measure it:** Instruments' Core ML template (Neural Engine track), Metal System Trace and the Thermal State track.

## 1. Platform

- **Hardware:** A12 Bionic (2 fast + 4 efficient CPU cores, 4-core GPU, 8-core Neural Engine), 4 GB RAM. It is not a Metal 3 device, so do not rely on Metal 3 features.
- **Software:** minimum iOS 18.0 (the XS Max's last release is 18.7.8), current Xcode, Swift 6 language mode, portrait only.
- **Dependencies:** none from third parties. Use Swift Package Manager if anything is needed. Drop TensorFlow Lite and CocoaPods.
- **Testing:** test on the XS Max itself. The Simulator has no camera and no Neural Engine, and is not representative.

## 2. Inference

- **Use Core ML, not TFLite.** Core ML is the supported path to the Neural Engine. TFLite's Core ML delegate only covers some operators.
- **Precision:** fp16, the Neural Engine's native type. Weight compression is optional and only worth it if measurements show a gain.
- **Model input:**
  - **Rectangular input:** for 16:9 camera frames, export with a rectangular size such as **640x352** (multiples of 32). It needs about 45% fewer operations than 640x640 and no padding.
  - **Sizes to compare:** benchmark 640x352, 416 and 320 against accuracy.
  - **Model families:** compare YOLOv5n/s with newer anchor-free models such as YOLOv8n/YOLO11n. Those have a different output layout with no objectness value, so the decoder must be model-specific.
- **Licensing:** Ultralytics models are AGPL-3.0. Decide this before committing to a model, especially if you will distribute the app.
- **Verify at load time:** check tensor shapes and class count, and refuse to load a mismatched model instead of crashing.
- **Confirm the model runs on the Neural Engine:** use `MLComputePlan` and Xcode's Core ML performance report. Operators that fall back to the CPU or GPU are the usual performance killers.
- **Compute units:** compare `.cpuAndNeuralEngine` with `.all` on the device.

## 3. Pipeline (this is what raises utilization)

- **Overlap the stages** so the Neural Engine never waits: prepare frame N+1 while frame N is running and frame N-1 is being decoded and drawn.
- **2-3 frames in flight**, using async predictions (`MLModel.prediction(from:) async`). Drop the newest frame if all slots are busy, and never queue frames.
- **No fixed throttle.** This project's current 200 ms cap limits detection to about 5 FPS. Let the pipeline run as fast as the model allows.
- **Zero allocations per frame in the steady state:**
  - **Buffers:** use a `CVPixelBufferPool`, preallocated buffers and `MLPredictionOptions.outputBackings`.
  - **Data types:** no `NSNumber` boxing and no `[Float]` copies. Use Accelerate/SIMD for any per-pixel or per-row work.
- **Concurrency:** actors or a dedicated serial executor for the pipeline. The model is never touched from two threads, and only the UI uses the main thread.
- **Postprocessing budget:** decode plus NMS stays under about 1.5 ms. Skip candidates below the threshold early. NMS is per class, with the IoU threshold separate from the confidence threshold.

## 4. Camera

- **Pick the `AVCaptureDevice.Format` explicitly:** 1280x720, with a frame rate chosen to match what the model sustains. Default presets are not tuned.
- **Native YUV pixel format** (`420v`/`420f`) if Vision or Core ML accepts it, so the ISP does not have to convert to BGRA.
- **Turn off extras that add latency or work:** video stabilization and video HDR.
- **Lock the frame rate** with `activeVideoMinFrameDuration` / `activeVideoMaxFrameDuration`, and set `alwaysDiscardsLateVideoFrames = true`.
- **Aspect ratio:** match the model's rectangular input to the frame to avoid resizing where possible. If resizing is needed, do it on the GPU or with vImage, not in a per-pixel loop.
- **Handle camera failure gracefully:** no camera, permission denied or restricted, and interruptions. No `fatalError`.

## 5. Display

- **Preview:** `AVCaptureVideoPreviewLayer` (GPU-composited, no CPU copy).
- **Overlay:** a pool of reused `CAShapeLayer`/`CATextLayer` objects, updated inside a `CATransaction` with implicit animations off. No full redraw per frame.
- **Coordinate mapping:** convert boxes with `layerRectConverted(fromMetadataOutputRect:)`. This handles aspect-fill correctly and avoids the box drift suspected in the current app.
- **Stats overlay:** update it at 2-4 Hz, not per frame. Use a dark, mostly black UI, which saves power on the OLED screen.

## 6. Thermal and power management

- **Watch `ProcessInfo.thermalState` and `isLowPowerModeEnabled`.** At `.fair` and above, step down in order: cap the frame rate, reduce detection frequency, switch to a smaller model or input size. Recover when the state drops back.
- **Idle timer:** disable it only while the camera is running.
- **Targets to validate:**
  - sustained detection for at least 15 minutes without staying at `.serious`;
  - end-to-end latency (camera frame to drawn box) of 100 ms or less;
  - steady memory with no growth.

## 7. Measurement and quality

- **Built-in benchmark mode:** per-stage timings (capture, preprocess, prediction, decode, draw), FPS, thermal state and memory, with `os_signpost` intervals for Instruments.
- **Test performance in Release only.** Debug builds distort every CPU-side number.
- **Unit tests:** letterbox and coordinate mapping, decode, NMS, and a golden-image test comparing Swift output with a Python reference.
- **XCTest performance tests** with signpost metrics.
- **Spike first:** before writing app code, build a small benchmark comparing the Neural Engine, GPU and the current TFLite Metal path on the real phone, at 640x352 / 416 / 320. Let the numbers pick the model and settings.

## 8. Lessons from this project

- **Keep:** letterbox math, per-class NMS, tensor-shape validation, drop-when-busy frames, per-stage timing.
- **Avoid:**
  - **Wrong preprocessing:** input normalization that does not match training (this project had `/127.5` instead of `/255`), and stretching frames instead of letterboxing.
  - **Threading:** loading models on the main thread, and sharing model state across threads.
  - **Per-frame cost:** per-frame table reloads and full overlay redraws.
  - **Testing in Debug:** performance tests must run in Release.

## Open decisions

1. What should it detect: COCO's 80 classes, or custom classes?
2. Which model license is acceptable?
3. Which matters more: maximum FPS, or long battery-friendly runs?
