

import UIKit

struct Prediction {
  let classIndex: Int
  let score: Float
  let rect: CGRect
}

class PrePostProcessor : NSObject {
    /// Minimum `objectness * class score` for a candidate box to be kept (YOLOv5's default is 0.25).
    static let confidenceThreshold: Float = 0.25
    /// Boxes of the same class overlapping a higher-scoring box by more than this IoU are suppressed.
    static let iouThreshold: Float = 0.45
    /// Maximum number of detections returned per frame.
    static let nmsLimit = 100
    /// Number of values preceding the class scores in each output row: x, y, w, h and objectness.
    static let boxValueCount = 5

    // The two methods nonMaxSuppression and IOU below are from  https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
      Removes bounding boxes that overlap too much with other boxes that have
      a higher score.
      - Parameters:
        - boxes: an array of bounding boxes and their scores
        - limit: the maximum number of boxes that will be selected
        - threshold: used to decide whether boxes overlap too much
    */
    static func nonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {
      // Do an argsort on the confidence scores, from high to low.
      let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

      var selected: [Prediction] = []
      var active = [Bool](repeating: true, count: boxes.count)
      var numActive = active.count

      // The algorithm is simple: Start with the box that has the highest score.
      // Remove any remaining boxes that overlap it more than the given threshold
      // amount. If there are any boxes left (i.e. these did not overlap with any
      // previous boxes), then repeat this procedure, until no more boxes remain
      // or the limit has been reached.
      outer: for i in 0..<boxes.count {
        if active[i] {
          let boxA = boxes[sortedIndices[i]]
          selected.append(boxA)
          if selected.count >= limit { break }

          for j in i+1..<boxes.count {
            if active[j] {
              let boxB = boxes[sortedIndices[j]]
              if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                active[j] = false
                numActive -= 1
                if numActive <= 0 { break outer }
              }
            }
          }
        }
      }
      return selected
    }

    /**
      Computes intersection-over-union overlap between two bounding boxes.
    */
    static func IOU(a: CGRect, b: CGRect) -> Float {
      let areaA = a.width * a.height
      if areaA <= 0 { return 0 }

      let areaB = b.width * b.height
      if areaB <= 0 { return 0 }

      let intersectionMinX = max(a.minX, b.minX)
      let intersectionMinY = max(a.minY, b.minY)
      let intersectionMaxX = min(a.maxX, b.maxX)
      let intersectionMaxY = min(a.maxY, b.maxY)
      let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                             max(intersectionMaxX - intersectionMinX, 0)
      return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }

    /// Runs NMS separately for each class so that overlapping objects of different classes do not
    /// suppress each other, then returns the best `limit` boxes overall.
    static func perClassNonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {
        let byClass = Dictionary(grouping: boxes, by: { $0.classIndex })
        var selected = [Prediction]()
        for (_, classBoxes) in byClass {
            selected += nonMaxSuppression(boxes: classBoxes, limit: limit, threshold: threshold)
        }
        selected.sort { $0.score > $1.score }
        return Array(selected.prefix(limit))
    }

    /// Decodes the raw YOLOv5 output into predictions expressed in source-image pixels.
    ///
    /// - Parameters:
    ///   - outputs: Flattened `[rows x columns]` output tensor. Each row is
    ///       `x, y, w, h, objectness, class scores...`, with the box normalised to the model input.
    ///   - rows: Number of candidate boxes in the output.
    ///   - columns: Number of values per candidate (`boxValueCount` + number of classes).
    ///   - inputSize: Size of the model input the frame was letterboxed into.
    ///   - letterbox: Scale and padding that were applied to fit the frame into the model input.
    ///   - imageWidth: Width of the original frame in pixels.
    ///   - imageHeight: Height of the original frame in pixels.
    static func outputsToNMSPredictions(outputs: UnsafeBufferPointer<Float>, rows: Int, columns: Int,
                                        inputSize: CGSize, letterbox: Letterbox,
                                        imageWidth: CGFloat, imageHeight: CGFloat) -> [Prediction] {
        let classCount = columns - boxValueCount
        guard classCount > 0, rows > 0, outputs.count >= rows * columns else { return [] }

        var predictions = [Prediction]()
        for i in 0..<rows {
            let base = i * columns
            let objectness = outputs[base + 4]
            // Cheap early-out: the combined score can never exceed the objectness.
            guard objectness > confidenceThreshold else { continue }

            var bestClassScore = outputs[base + boxValueCount]
            var cls = 0
            for j in 1..<max(classCount, 1) {
                let classScore = outputs[base + boxValueCount + j]
                if classScore > bestClassScore {
                    bestClassScore = classScore
                    cls = j
                }
            }

            let score = objectness * bestClassScore
            guard score >= confidenceThreshold else { continue }

            // Normalised centre/size -> model-input pixels -> source-image pixels (undoing the letterbox).
            let x = CGFloat(outputs[base]) * inputSize.width
            let y = CGFloat(outputs[base + 1]) * inputSize.height
            let w = CGFloat(outputs[base + 2]) * inputSize.width
            let h = CGFloat(outputs[base + 3]) * inputSize.height

            let left = ((x - w / 2) - letterbox.padX) / letterbox.scale
            let top = ((y - h / 2) - letterbox.padY) / letterbox.scale
            let right = ((x + w / 2) - letterbox.padX) / letterbox.scale
            let bottom = ((y + h / 2) - letterbox.padY) / letterbox.scale

            let clampedLeft = min(max(left, 0), imageWidth)
            let clampedTop = min(max(top, 0), imageHeight)
            let clampedRight = min(max(right, 0), imageWidth)
            let clampedBottom = min(max(bottom, 0), imageHeight)
            guard clampedRight > clampedLeft, clampedBottom > clampedTop else { continue }

            let rect = CGRect(x: clampedLeft, y: clampedTop,
                              width: clampedRight - clampedLeft, height: clampedBottom - clampedTop)
            predictions.append(Prediction(classIndex: cls, score: score, rect: rect))
        }

        return perClassNonMaxSuppression(boxes: predictions, limit: nmsLimit, threshold: iouThreshold)
    }

    static func cleanDetection(imageView: UIImageView) {
        if let layers = imageView.layer.sublayers {
            for layer in layers {
                if layer is CATextLayer {
                    layer.removeFromSuperlayer()
                }
            }
            for view in imageView.subviews {
                view.removeFromSuperview()
            }
        }
    }

    static func showDetection(imageView: UIImageView, nmsPredictions: [Prediction], classes: [String]) {
        
        for pred in nmsPredictions {
            let bbox = UIView(frame: pred.rect)
            bbox.backgroundColor = UIColor.clear
            bbox.layer.borderColor = UIColor.yellow.cgColor
            bbox.layer.borderWidth = 2
            imageView.addSubview(bbox)
            
            let textLayer = CATextLayer()
            textLayer.string = String(format: " %@ %.2f", classes[pred.classIndex], pred.score)
            textLayer.foregroundColor = UIColor.white.cgColor
            textLayer.backgroundColor = UIColor.magenta.cgColor
            textLayer.fontSize = 14
            textLayer.frame = CGRect(x: pred.rect.origin.x, y: pred.rect.origin.y, width:100, height:20)
            imageView.layer.addSublayer(textLayer)
            
        }
    }

}
