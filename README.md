# Fiber ID — fabric fiber classifier app

A Flutter app that photographs a piece of fabric and predicts its fiber —
**Cotton / Polyester / Silk** — entirely on-device, using a fine-tuned
MobileNetV2 model exported to TensorFlow Lite.

## Quick start

```bash
flutter pub get
flutter run
```

Point it at a connected device or emulator; the model is already bundled in
`assets/`, so no extra setup is needed to try it. If instead you're setting
this project up from scratch (no `android/`/`ios/` platform folders yet), see
[Full environment setup](#full-environment-setup) below.

## Model performance

Evaluated with the TFLite interpreter (the same one the app uses) on a
held-out test split the model never saw during training or validation.

| Split | Cotton | Polyester | Silk | Total |
|---|---|---|---|---|
| Train | 837 | 783 | 609 | 2,229 |
| Validation | 59 | 56 | 44 | 159 |
| Test | 60 | 56 | 43 | 159 |

**Test accuracy: 70.4%** (3-class, so well above the 33% random baseline, but
not high enough to trust blindly — treat predictions as a strong hint, not a
verdict).

Training curve (`train_fiber_classifier.py`):

| Stage | Epochs | End-of-stage train acc | End-of-stage val acc |
|---|---|---|---|
| Stage 1 — head only, MobileNetV2 frozen | 8 | ~82% | ~72% |
| Stage 2 — fine-tune top MobileNetV2 layers | 6 | ~83% | ~72% (flat) |

Validation accuracy plateaued in the low-to-mid 70s and didn't move during
fine-tuning while train accuracy kept climbing — a sign of mild overfitting
given only ~2,200 training images spread across 3 classes. Cotton and
Polyester are the more reliably distinguished pair; Silk has the fewest
examples and the most visual overlap with the others. More data (especially
more Silk) or heavier augmentation would be the first lever to pull if
accuracy needs to improve.

## Dataset

Source: [Fabric Classification](https://universe.roboflow.com/fabric-classification-01qgr/fabric-classification-uimqb)
dataset on Roboflow Universe (CC BY 4.0), exported in folder-classification
format with images resized to 224×224 and offline-augmented 3x (horizontal
flip, ±10° rotation, ±15% brightness) by Roboflow at export time.

The original export had 7 classes (Cotton, Polyester, Silk, Denim, Wool,
Viscose, Cotton-Mixed). This project only trains on **Cotton / Polyester /
Silk** — the other four classes were moved to `dataset_excluded/` and are not
part of the training run. That means: **photographing denim, wool, or any
other fabric will still get force-classified as one of the three trained
classes** — there's no "none of the above" option.

`dataset/` layout expected by the training script:

```
dataset/
  train/{Cotton,Polyester,Silk}/*.jpg
  valid/{Cotton,Polyester,Silk}/*.jpg
  test/{Cotton,Polyester,Silk}/*.jpg
```

## Model architecture

Defined in `train_fiber_classifier.py`, transfer-learned from ImageNet:

```
Input (224×224×3, raw 0-255 pixels)
  -> RandomFlip(horizontal) + RandomRotation(0.05)      [train-time only]
  -> preprocess_input (MobileNetV2 scaling, 0-255 -> -1..1)   [baked into graph]
  -> MobileNetV2(include_top=False, weights=imagenet)   [backbone]
  -> GlobalAveragePooling2D
  -> Dropout(0.2)
  -> Dense(3, activation=softmax)
```

Trained in two stages:

1. **Head training** (8 epochs, `Adam(lr=1e-3)`) — MobileNetV2 backbone
   entirely frozen, only the new pooling/dropout/dense head is trained.
2. **Fine-tuning** (6 epochs, `Adam(lr=1e-5)`) — backbone layers from index
   100 onward are unfrozen and trained at a much lower learning rate; layers
   before index 100 (the earlier, more generic feature extractors) stay
   frozen.

The MobileNetV2 `preprocess_input` normalization is baked directly into the
Keras graph (as a `Lambda` layer) before conversion to TFLite, so the app can
feed raw decoded pixel values straight in — `classifier.dart` does not
re-normalize. If you ever move that normalization out of the model, you must
update `_preprocess` in `classifier.dart` to match, or predictions will be
silently wrong.

Exported model: `fiber_classifier.tflite`, 8.9 MB. Interpreter shapes:
- Input: `[1, 224, 224, 3]` float32
- Output: `[1, 3]` float32 (softmax over `labels.txt` order: Cotton,
  Polyester, Silk)

## Retraining the model

1. Put a Roboflow-style folder-classification export at `./dataset` (see
   layout above; `dataset/README.roboflow.txt` has the original export notes).
2. `pip install tensorflow-cpu`
3. `python train_fiber_classifier.py` — trains both stages, evaluates on
   `dataset/test`, and writes `fiber_classifier.tflite` + `labels.txt` to the
   project root.
4. Copy both files into `assets/` (overwriting the existing ones).
5. `flutter pub get` and re-run — no code changes needed as long as the
   number of classes and their order in `labels.txt` stay the same shape the
   app expects.

## Full environment setup

If you're cloning this repo fresh, in order:

1. **Flutter SDK** — install from [flutter.dev](https://docs.flutter.dev/get-started/install)
   and make sure `flutter` is on your `PATH`.
2. **JDK 17** — Android Gradle builds need it (JDK 17, not 21+). Set
   `JAVA_HOME` to it.
3. **Android SDK** — via Android Studio, or the standalone `cmdline-tools` +
   `sdkmanager`. You need at least one `platforms;android-XX` and
   `build-tools` installed, plus `platform-tools` (for `adb`).
4. Run `flutter doctor` and resolve anything it flags (accept Android
   licenses with `flutter doctor --android-licenses` if prompted).
5. From the project root:
   ```bash
   flutter create .   # only if android/, ios/, etc. don't already exist
   flutter pub get
   flutter run        # with a device connected (USB debugging enabled) or an emulator running
   ```

## Permissions

- **Android** — `android/app/src/main/AndroidManifest.xml` already requests
  `android.permission.CAMERA`.
- **iOS** — before shipping, add to `ios/Runner/Info.plist`:
  ```xml
  <key>NSCameraUsageDescription</key>
  <string>Photograph fabric to identify the fiber.</string>
  <key>NSPhotoLibraryUsageDescription</key>
  <string>Pick a fabric photo to identify the fiber.</string>
  ```

## How the pieces fit

- `lib/classifier.dart` — loads the `.tflite` model, resizes the input image
  to 224×224, runs inference, returns confidences sorted descending.
- `lib/main.dart` — camera/gallery picker, image preview, confidence bars UI.
- `assets/labels.txt` — class order; must exactly match the order the model
  was trained with (index 0 = first line, etc.).
- `train_fiber_classifier.py` — training/export script, see above.
