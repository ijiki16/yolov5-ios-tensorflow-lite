import Foundation
import os

/// Thin wrapper over `os_log` so messages show up in Console.app and Xcode's log, tagged with the
/// app's subsystem, instead of being written with `print`.
enum Log {
  private static let logger = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "ObjectDetection",
                                    category: "ObjectDetection")

  static func info(_ message: String) {
    os_log("%{public}@", log: logger, type: .info, message)
  }

  static func error(_ message: String) {
    os_log("%{public}@", log: logger, type: .error, message)
  }
}
