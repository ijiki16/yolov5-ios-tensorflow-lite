// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import Foundation
import Accelerate

/// Describes how a source image was fitted into the model input, so that model-space coordinates
/// can be mapped back to source-image coordinates.
struct Letterbox {
  /// Factor by which the source image was scaled to fit the model input.
  let scale: CGFloat
  /// Horizontal padding (in model-input pixels) added on the left side.
  let padX: CGFloat
  /// Vertical padding (in model-input pixels) added on the top side.
  let padY: CGFloat

  /// Value used for the padding channels, matching YOLOv5's own letterbox padding colour.
  static let paddingValue: UInt8 = 114

  init(scale: CGFloat, padX: CGFloat, padY: CGFloat) {
    self.scale = scale
    self.padX = padX
    self.padY = padY
  }

  init(sourceWidth: Int, sourceHeight: Int, targetWidth: Int, targetHeight: Int) {
    scale = min(CGFloat(targetWidth) / CGFloat(sourceWidth), CGFloat(targetHeight) / CGFloat(sourceHeight))
    padX = (CGFloat(targetWidth) - CGFloat(sourceWidth) * scale) / 2
    padY = (CGFloat(targetHeight) - CGFloat(sourceHeight) * scale) / 2
  }

  /// Size of the scaled image (without padding), in model-input pixels.
  func scaledSize(sourceWidth: Int, sourceHeight: Int) -> (width: Int, height: Int) {
    return (max(1, Int((CGFloat(sourceWidth) * scale).rounded())),
            max(1, Int((CGFloat(sourceHeight) * scale).rounded())))
  }
}

extension CVPixelBuffer {
  /// Scales the pixel buffer to fit inside `size` while preserving its aspect ratio, centres it,
  /// and fills the remaining area with grey padding (a "letterbox"). Returns the resulting buffer
  /// together with the parameters needed to map model coordinates back to this buffer.
  func letterboxed(to size: CGSize) -> (buffer: CVPixelBuffer, letterbox: Letterbox)? {

    let imageWidth = CVPixelBufferGetWidth(self)
    let imageHeight = CVPixelBufferGetHeight(self)
    let targetWidth = Int(size.width)
    let targetHeight = Int(size.height)

    let pixelBufferType = CVPixelBufferGetPixelFormatType(self)
    guard pixelBufferType == kCVPixelFormatType_32BGRA ||
          pixelBufferType == kCVPixelFormatType_32ARGB,
          imageWidth > 0, imageHeight > 0, targetWidth > 0, targetHeight > 0 else {
      return nil
    }

    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
    let imageChannels = 4

    CVPixelBufferLockBaseAddress(self, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }

    guard let inputBaseAddress = CVPixelBufferGetBaseAddress(self) else {
      return nil
    }

    var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress, height: UInt(imageHeight), width: UInt(imageWidth), rowBytes: inputImageRowBytes)

    let letterbox = Letterbox(sourceWidth: imageWidth, sourceHeight: imageHeight,
                              targetWidth: targetWidth, targetHeight: targetHeight)
    let scaled = letterbox.scaledSize(sourceWidth: imageWidth, sourceHeight: imageHeight)
    // Pixel offsets are rounded down; the size is clamped so the scaled image always fits.
    let offsetX = min(Int(letterbox.padX), targetWidth - scaled.width)
    let offsetY = min(Int(letterbox.padY), targetHeight - scaled.height)

    let outputRowBytes = targetWidth * imageChannels
    guard let outputBytes = malloc(targetHeight * outputRowBytes) else {
      return nil
    }

    // Fill the whole output with the padding colour (alpha stays opaque).
    let padding = Letterbox.paddingValue
    var fill: [UInt8] = [padding, padding, padding, 255]
    memset_pattern4(outputBytes, &fill, targetHeight * outputRowBytes)

    // Scale the source straight into the centred sub-rectangle of the output.
    let regionStart = outputBytes + offsetY * outputRowBytes + offsetX * imageChannels
    var regionVImageBuffer = vImage_Buffer(data: regionStart, height: UInt(scaled.height), width: UInt(scaled.width), rowBytes: outputRowBytes)
    let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &regionVImageBuffer, nil, vImage_Flags(0))

    guard scaleError == kvImageNoError else {
      free(outputBytes)
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = { _, pointer in
      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }

    var outputPixelBuffer: CVPixelBuffer?
    let conversionStatus = CVPixelBufferCreateWithBytes(nil, targetWidth, targetHeight, pixelBufferType, outputBytes, outputRowBytes, releaseCallBack, nil, nil, &outputPixelBuffer)

    guard conversionStatus == kCVReturnSuccess, let result = outputPixelBuffer else {
      free(outputBytes)
      return nil
    }

    // Use the offsets actually applied, which may differ from the ideal padding by under a pixel.
    let applied = Letterbox(scale: letterbox.scale, padX: CGFloat(offsetX), padY: CGFloat(offsetY))
    return (result, applied)
  }
}
