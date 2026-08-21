# Fiber ID — fabric fiber classifier app

A Flutter app that photographs fabric and predicts **cotton / polyester / silk**
on-device. Runs today in *demo mode*; drop in a trained model later for real
predictions.

## Run it now (no model needed)

```bash
flutter create .          # generates android/ ios/ platform folders in this dir
flutter pub get
flutter run
```

The app launches in **demo mode** (random results, shown with a banner) so you
can wire up UI and camera flow before the model exists.

## Add the trained model

1. Put a Roboflow folder-classification export at `./dataset` (train/valid/test,
   one subfolder per class — see `dataset/README.roboflow.txt`).
2. Run `pip install tensorflow-cpu` then `python train_fiber_classifier.py` —
   it produces `fiber_classifier.tflite` and `labels.txt`.
3. Copy `fiber_classifier.tflite` into `assets/`.
4. In `pubspec.yaml`, uncomment the model asset line:
   ```yaml
   - assets/fiber_classifier.tflite
   ```
5. `flutter pub get` and re-run. The demo banner disappears and predictions
   become real.

## Permissions

Add camera/photo permissions before shipping:

- **Android** — in `android/app/src/main/AndroidManifest.xml`:
  ```xml
  <uses-permission android:name="android.permission.CAMERA"/>
  ```
- **iOS** — in `ios/Runner/Info.plist`:
  ```xml
  <key>NSCameraUsageDescription</key>
  <string>Photograph fabric to identify the fiber.</string>
  <key>NSPhotoLibraryUsageDescription</key>
  <string>Pick a fabric photo to identify the fiber.</string>
  ```

## How the pieces fit

- `lib/classifier.dart` — loads the `.tflite`, resizes the image to 224×224,
  runs inference, returns sorted confidences. Falls back to demo mode if the
  model asset is missing.
- `lib/main.dart` — camera/gallery picker, image preview, confidence bars.
- `assets/labels.txt` — class order; **must match** the training script.

## Important: preprocessing must match training

The training script bakes MobileNetV2's `preprocess_input` (scale to [-1, 1])
*inside* the model, so the app feeds raw 0–255 pixels. If you ever move that
normalization out of the model, update `_preprocess` in `classifier.dart` to
match — mismatched preprocessing is the #1 cause of "works in Python, garbage in
the app."
