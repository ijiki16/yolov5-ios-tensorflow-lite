import UIKit

/// Owns the window for a scene on iOS 13 and later. The window and its root view controller are
/// created from the `Main` storyboard named in Info.plist (`UISceneStoryboardFile`), so nothing
/// else needs to be set up here. On iOS 12, `AppDelegate` and `UIMainStoryboardFile` are used instead.
@available(iOS 13.0, *)
class SceneDelegate: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?
}
