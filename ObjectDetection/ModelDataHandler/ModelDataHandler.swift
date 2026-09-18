// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
    let inferenceTime: Double
    let inferences: [Inference]
}

/// Stores one formatted inference.
struct Inference {
    let confidence: Float
    let className: String
    let rect: CGRect
    let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum Yolov5 {
    static let modelInfo: FileInfo = (name: "yolov5s-fp16", extension: "tflite")
    static let labelsInfo: FileInfo = (name: "classes", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {
    
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int
    let threadCountLimit = 10
    
    // MARK: Model parameters
    let batchSize = 1
    let inputChannels = 3
    /// Input size, read from the model's input tensor shape (`[1, height, width, 3]`).
    let inputWidth: Int
    let inputHeight: Int
    /// Output layout, read from the model's output tensor shape (`[1, rows, columns]`).
    private let outputRows: Int
    private let outputColumns: Int
    
    // Image mean and std for floating models. YOLOv5 TFLite exports expect pixels scaled to [0, 1].
    let imageMean: Float = 0
    let imageStd:  Float = 255
    
    // MARK: Private properties
    private var labels: [String] = []
    
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter
    
    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    private let colorStrideValue = 10
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown
    ]
    
    // MARK: - Initialization
    
    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
        let modelFilename = modelFileInfo.name
        
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = Interpreter.Options()
        options.threadCount = threadCount
        let inputDimensions: [Int]
        let outputDimensions: [Int]
        do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
            inputDimensions = try interpreter.input(at: 0).shape.dimensions
            outputDimensions = try interpreter.output(at: 0).shape.dimensions
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Expect NHWC input `[1, height, width, 3]` and output `[1, rows, 5 + classCount]`.
        guard inputDimensions.count == 4, inputDimensions[0] == batchSize, inputDimensions[3] == inputChannels,
              outputDimensions.count == 3, outputDimensions[0] == batchSize,
              outputDimensions[2] > PrePostProcessor.boxValueCount else {
            print("Unexpected model shapes: input \(inputDimensions), output \(outputDimensions).")
            return nil
        }
        inputHeight = inputDimensions[1]
        inputWidth = inputDimensions[2]
        outputRows = outputDimensions[1]
        outputColumns = outputDimensions[2]
        
        super.init()
        
        // Load the classes listed in the labels file.
        loadLabels(fileInfo: labelsFileInfo)
        
        // Every class the model can predict needs a label.
        guard labels.count >= outputColumns - PrePostProcessor.boxValueCount else {
            print("Model predicts \(outputColumns - PrePostProcessor.boxValueCount) classes but only " +
                  "\(labels.count) labels were loaded.")
            return nil
        }
    }
    
    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        // Scales the image to fit the model input without distorting it, padding the remainder.
        let inputSize = CGSize(width: inputWidth, height: inputHeight)
        guard let (scaledPixelBuffer, letterbox) = pixelBuffer.letterboxed(to: inputSize) else {
            print("Failed to resize the frame; only 32BGRA and 32ARGB pixel buffers are supported.")
            return nil
        }
        
        let interval: TimeInterval
        let outputResult: Tensor

        do {
            let inputTensor = try interpreter.input(at: 0)
            
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
            
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            let startTime = CACurrentMediaTime()
            try interpreter.invoke()
            interval = (CACurrentMediaTime() - startTime) * 1000
            
            outputResult = try interpreter.output(at: 0)
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Decode straight from the tensor's bytes to avoid copying ~2M floats into an array.
        let outputData = outputResult.data
        let expectedByteCount = outputRows * outputColumns * MemoryLayout<Float>.stride
        guard outputData.count >= expectedByteCount else {
            print("Unexpected output tensor size.")
            return nil
        }
        let nmsPredictions = outputData.withUnsafeBytes { rawBuffer in
            PrePostProcessor.outputsToNMSPredictions(
                outputs: rawBuffer.bindMemory(to: Float.self), rows: outputRows, columns: outputColumns,
                inputSize: inputSize, letterbox: letterbox,
                imageWidth: CGFloat(imageWidth), imageHeight: CGFloat(imageHeight))
        }

        var inference: [Inference] = []
        for prediction in nmsPredictions {
            let pred = Inference(confidence: prediction.score, className: labels[prediction.classIndex], rect: prediction.rect, displayColor: colorForClass(withIndex: prediction.classIndex + 1))
            inference.append(pred)
        }
        let result = Result(inferenceTime: interval, inferences: inference)

        return result
    }
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
    private func loadLabels(fileInfo: FileInfo) {
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            fatalError("Labels file not found in bundle. Please add a labels file with name " +
                       "\(filename).\(fileExtension) and try again.")
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labels = contents.components(separatedBy: .newlines)
        } catch {
            fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                       "valid labels file and try again.")
        }
    }
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        }
        
        let byteCount = destinationBuffer.rowBytes * height
        if isModelQuantized {
            return Data(bytes: destinationData, count: byteCount)
        }
        
        // Not quantized: convert to floats and normalise as `(x - mean) / std` using vDSP.
        var floats = [Float](repeating: 0, count: byteCount)
        var scale = 1 / imageStd
        var offset = -imageMean / imageStd
        floats.withUnsafeMutableBufferPointer { floatBuffer in
            guard let floatBase = floatBuffer.baseAddress else { return }
            vDSP_vfltu8(destinationData.assumingMemoryBound(to: UInt8.self), 1,
                        floatBase, 1, vDSP_Length(byteCount))
            vDSP_vsmsa(floatBase, 1, &scale, &offset, floatBase, 1, vDSP_Length(byteCount))
        }
        return Data(copyingBufferOf: floats)
    }
    
    /// This assigns color for a particular class.
    private func colorForClass(withIndex index: Int) -> UIColor {
        
        // We have a set of colors and the depending upon a stride, it assigns variations to of the base
        // colors to each object based on its index.
        let baseColor = colors[index % colors.count]
        
        var colorToAssign = baseColor
        
        let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)
        
        if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
            colorToAssign = modifiedColor
        }
        
        return colorToAssign
    }
}

// MARK: - Extensions

extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    ///
    /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
    ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
    ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
    /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
    ///     `MemoryLayout<Element>.stride`.
    /// - Parameter unsafeData: The data containing the bytes to turn into an array.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
#if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
#else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
#endif  // swift(>=5.0)
    }
}
