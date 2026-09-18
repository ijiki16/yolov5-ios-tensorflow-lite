# Yolov5 iOS TensorflowLite

[English README](README.md) | [ქართული README](README_georgian.md)

TensorFlow Lite로 변환한 [YOLOv5](https://github.com/ultralytics/yolov5) 모델로 실시간 객체 검출을 수행하는 iOS 예제 앱입니다. 후면 카메라 영상에서 바운딩 박스, 클래스, 신뢰도를 표시합니다. TensorFlow Lite Metal delegate로 GPU에서 실행되며, delegate를 만들 수 없으면 CPU로 자동 전환됩니다.

**지원 iOS 버전:** iOS 12.0 이상

## 빌드 방법

1. CocoaPods를 설치합니다: `sudo gem install cocoapods`
2. 프로젝트 폴더에서 `pod install`을 실행합니다. 처음 실행할 때는 TensorFlow 저장소를 clone하므로 시간이 걸릴 수 있습니다.
3. 서명 설정을 준비합니다. 예시 파일을 복사한 뒤 본인의 Team ID와 고유한 Bundle Identifier를 입력하세요.

   ```sh
   cp Config/Local.xcconfig.example Config/Local.xcconfig
   ```

   `Config/Local.xcconfig`는 git에서 무시되므로 개인 서명 정보가 커밋되지 않습니다.
4. `ObjectDetection.xcworkspace`를 Xcode로 열고 실제 iOS 기기에서 실행합니다. 카메라가 필요하므로 시뮬레이터에서는 검출이 동작하지 않습니다 (실행은 되지만 "Configuration Failed" 알림이 표시됩니다).

## 모델

`ObjectDetection/Model`에 COCO로 학습된 두 모델이 포함되어 있습니다.

| 파일 | 설명 |
| --- | --- |
| `yolov5s-fp16.tflite` | 더 정확하지만 느립니다. 기본값입니다. |
| `yolov5n-fp16.tflite` | 덜 정확하지만 더 빠릅니다. |

모델을 바꾸려면 `ModelDataHandler.swift`의 `Yolov5.modelInfo`를 `Yolov5.nanoModelInfo`로 변경하세요.

- **입력:** `[1, height, width, 3]`, RGB, `[0, 1]`로 정규화된 float. 기본 모델은 `640 x 640`이며, 비율을 유지한 채 회색으로 패딩(letterbox)해서 입력합니다.
- **출력:** `[1, rows, 5 + 클래스 수]`. 기본 모델은 `[1, 25200, 85]` (COCO 80개 클래스)입니다.

### 직접 학습한 모델 사용

1. [YOLOv5 저장소](https://github.com/ultralytics/yolov5)의 `export.py`로 tflite 모델을 만듭니다.

   ```sh
   python export.py --weights your_model.pt --include tflite
   ```
2. `.tflite` 파일을 `ObjectDetection/Model`에 추가하고 Xcode 타깃에 포함시킵니다.
3. `classes.txt`를 학습 때와 같은 순서의 클래스 이름(한 줄에 하나)으로 교체합니다.
4. `Yolov5.modelInfo`가 새 모델을 가리키도록 변경합니다.

### 검출 임계값

`PrePostProcessor.swift`에서 조절할 수 있습니다.

- `confidenceThreshold` (기본 `0.25`): 검출로 인정할 최소 `objectness * 클래스 점수`
- `iouThreshold` (기본 `0.45`): 같은 클래스의 박스가 이 값 이상 겹치면 점수가 낮은 박스를 제거
- `nmsLimit` (기본 `100`): 프레임당 최대 검출 개수

## 라이선스

이 앱은 [TensorFlow examples](https://github.com/tensorflow/examples)의 객체 검출 예제를 기반으로 하며, Apache License 2.0으로 배포됩니다. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
