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
struct InferenceResult {
    /// Time spent in `Interpreter.invoke()`, in milliseconds.
    let inferenceTime: Double
    /// Time spent resizing/converting the frame and filling the input tensor, in milliseconds.
    let preprocessTime: Double
    /// Time spent reading the output tensor back, decoding it and running NMS, in milliseconds.
    let postprocessTime: Double
    /// Whether this frame ran on the GPU (Metal delegate) rather than the CPU.
    let isUsingGPU: Bool
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

/// Information about the bundled YOLOv5 models and their labels.
enum Yolov5 {
    /// Small model: more accurate, slower.
    static let smallModelInfo: FileInfo = (name: "yolov5s-fp16", extension: "tflite")
    /// Nano model: less accurate, faster.
    static let nanoModelInfo: FileInfo = (name: "yolov5n-fp16", extension: "tflite")
    /// The model the app runs. Switch to `nanoModelInfo` for a lighter model.
    static let modelInfo: FileInfo = smallModelInfo
    static let labelsInfo: FileInfo = (name: "classes", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {
    
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter when running on the CPU.
    let threadCount: Int
    let threadCountLimit = 10
    /// Whether inference runs on the GPU through the Metal delegate (otherwise on the CPU).
    let isUsingGPU: Bool
    
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
    ///
    /// When `useGPU` is true the Metal delegate is tried first, falling back to the CPU if it cannot
    /// be created; `threadCount` then only matters for the CPU path.
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1, useGPU: Bool = true) {
        let modelFilename = modelFileInfo.name
        
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            Log.error("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        self.threadCount = threadCount
        let inputDimensions: [Int]
        let outputDimensions: [Int]
        do {
            // Create the `Interpreter` (on the GPU if possible) and allocate the model's tensors.
            (interpreter, isUsingGPU) = try ModelDataHandler.makeInterpreter(
                modelPath: modelPath, threadCount: threadCount, useGPU: useGPU)
            inputDimensions = try interpreter.input(at: 0).shape.dimensions
            outputDimensions = try interpreter.output(at: 0).shape.dimensions
        } catch let error {
            Log.error("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Expect NHWC input `[1, height, width, 3]` and output `[1, rows, 5 + classCount]`.
        guard inputDimensions.count == 4, inputDimensions[0] == batchSize, inputDimensions[3] == inputChannels,
              outputDimensions.count == 3, outputDimensions[0] == batchSize,
              outputDimensions[2] > PrePostProcessor.boxValueCount else {
            Log.error("Unexpected model shapes: input \(inputDimensions), output \(outputDimensions).")
            return nil
        }
        inputHeight = inputDimensions[1]
        inputWidth = inputDimensions[2]
        outputRows = outputDimensions[1]
        outputColumns = outputDimensions[2]
        
        super.init()
        
        // Load the classes listed in the labels file.
        guard loadLabels(fileInfo: labelsFileInfo) else {
            return nil
        }
        
        // Every class the model can predict needs a label.
        guard labels.count >= outputColumns - PrePostProcessor.boxValueCount else {
            Log.error("Model predicts \(outputColumns - PrePostProcessor.boxValueCount) classes but only " +
                  "\(labels.count) labels were loaded.")
            return nil
        }
    }
    
    /// Creates an interpreter with its tensors allocated. If `useGPU` is true, the Metal delegate is
    /// tried first; any failure to set it up falls back to a CPU interpreter.
    /// - Returns: The interpreter and whether it is using the GPU.
    private static func makeInterpreter(modelPath: String, threadCount: Int, useGPU: Bool) throws
        -> (Interpreter, Bool) {
        if useGPU {
            do {
                var metalOptions = MetalDelegate.Options()
                // fp16 precision is what the GPU is fastest at, and is plenty for detection.
                metalOptions.isPrecisionLossAllowed = true
                let interpreter = try Interpreter(modelPath: modelPath, delegates: [MetalDelegate(options: metalOptions)])
                try interpreter.allocateTensors()
                return (interpreter, true)
            } catch let error {
                Log.info("Metal delegate unavailable, falling back to the CPU: \(error.localizedDescription)")
            }
        }

        var options = Interpreter.Options()
        options.threadCount = threadCount
        let interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()
        return (interpreter, false)
    }

    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    ///
    /// Each stage is wrapped in a signpost interval (visible in Instruments' Points of Interest track)
    /// and timed, so the time per stage can be shown in the UI.
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> InferenceResult? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let inputSize = CGSize(width: inputWidth, height: inputHeight)
        
        // Stage 1: fit the frame into the model input, convert it to RGB floats and fill the input tensor.
        let (preparedLetterbox, preprocessTime) = Log.timed("Preprocess") { () -> Letterbox? in
            // Scales the image to fit the model input without distorting it, padding the remainder.
            guard let (scaledPixelBuffer, letterbox) = pixelBuffer.letterboxed(to: inputSize) else {
                Log.error("Failed to resize the frame; only 32BGRA and 32ARGB pixel buffers are supported.")
                return nil
            }
            do {
                let inputTensor = try interpreter.input(at: 0)
                
                // Remove the alpha component from the image buffer to get the RGB data.
                guard let rgbData = rgbDataFromBuffer(
                    scaledPixelBuffer,
                    byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                    isModelQuantized: inputTensor.dataType == .uInt8
                ) else {
                    Log.error("Failed to convert the image buffer to RGB data.")
                    return nil
                }
                
                // Copy the RGB data to the input `Tensor`.
                try interpreter.copy(rgbData, toInputAt: 0)
                return letterbox
            } catch let error {
                Log.error("Failed to prepare the input tensor with error: \(error.localizedDescription)")
                return nil
            }
        }
        guard let letterbox = preparedLetterbox else { return nil }
        
        // Stage 2: run inference by invoking the `Interpreter`.
        let (invokeSucceeded, inferenceTime) = Log.timed("Inference") { () -> Bool in
            do {
                try interpreter.invoke()
                return true
            } catch let error {
                Log.error("Failed to invoke the interpreter with error: \(error.localizedDescription)")
                return false
            }
        }
        guard invokeSucceeded else { return nil }
        
        // Stage 3: read the output back, decode it into boxes and suppress overlaps.
        let (decodedInferences, postprocessTime) = Log.timed("Postprocess") { () -> [Inference]? in
            let outputData: Data
            do {
                outputData = try interpreter.output(at: 0).data
            } catch let error {
                Log.error("Failed to read the output tensor with error: \(error.localizedDescription)")
                return nil
            }
            
            // Decode straight from the tensor's bytes to avoid copying ~2M floats into an array.
            let expectedByteCount = outputRows * outputColumns * MemoryLayout<Float>.stride
            guard outputData.count >= expectedByteCount else {
                Log.error("Unexpected output tensor size.")
                return nil
            }
            let nmsPredictions = outputData.withUnsafeBytes { rawBuffer in
                PrePostProcessor.outputsToNMSPredictions(
                    outputs: rawBuffer.bindMemory(to: Float.self), rows: outputRows, columns: outputColumns,
                    inputSize: inputSize, letterbox: letterbox,
                    imageWidth: CGFloat(imageWidth), imageHeight: CGFloat(imageHeight))
            }
            
            return nmsPredictions.map { prediction in
                Inference(confidence: prediction.score, className: labels[prediction.classIndex], rect: prediction.rect, displayColor: colorForClass(withIndex: prediction.classIndex + 1))
            }
        }
        guard let inferences = decodedInferences else { return nil }
        
        return InferenceResult(inferenceTime: inferenceTime, preprocessTime: preprocessTime,
                               postprocessTime: postprocessTime, isUsingGPU: isUsingGPU, inferences: inferences)
    }
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
    /// - Returns: `false` if the file is missing or cannot be read.
    private func loadLabels(fileInfo: FileInfo) -> Bool {
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            Log.error("Labels file not found in bundle. Please add a labels file with name " +
                      "\(filename).\(fileExtension) and try again.")
            return false
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labels = contents.components(separatedBy: .newlines)
            return true
        } catch {
            Log.error("Labels file named \(filename).\(fileExtension) cannot be read: \(error.localizedDescription)")
            return false
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
            Log.error("Error: out of memory")
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
