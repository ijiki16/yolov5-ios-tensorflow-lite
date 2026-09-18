import Foundation
import QuartzCore
import os

/// Thin wrapper over `os_log` so messages show up in Console.app and Xcode's log, tagged with the
/// app's subsystem, instead of being written with `print`.
enum Log {
  private static let logger = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "ObjectDetection",
                                    category: "ObjectDetection")

  /// Signposts show up as intervals in Instruments' "Points of Interest" track.
  private static let signpostLog = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "ObjectDetection",
                                         category: .pointsOfInterest)

  /// Runs `body` inside a signpost interval called `name` and returns its result along with how long
  /// it took, in milliseconds (measured with the monotonic media clock).
  static func timed<T>(_ name: StaticString, _ body: () throws -> T) rethrows -> (result: T, milliseconds: Double) {
    let signpostID = OSSignpostID(log: signpostLog)
    os_signpost(.begin, log: signpostLog, name: name, signpostID: signpostID)
    let start = CACurrentMediaTime()
    defer { os_signpost(.end, log: signpostLog, name: name, signpostID: signpostID) }
    let result = try body()
    return (result, (CACurrentMediaTime() - start) * 1000)
  }

  static func info(_ message: String) {
    os_log("%{public}@", log: logger, type: .info, message)
  }

  static func error(_ message: String) {
    os_log("%{public}@", log: logger, type: .error, message)
  }
}
