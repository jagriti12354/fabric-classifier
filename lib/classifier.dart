import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// One prediction row: a fiber name and the model's confidence (0..1).
class Prediction {
  final String label;
  final double confidence;
  const Prediction(this.label, this.confidence);
}

/// Wraps the TFLite model. If the model asset isn't bundled yet, it runs in
/// [demoMode] and returns clearly-fake results so the app still works before
/// you've trained anything.
class FiberClassifier {
  static const _modelAsset = 'assets/fiber_classifier.tflite';
  static const _labelsAsset = 'assets/labels.txt';
  static const _inputSize = 224; // must match training IMG_SIZE

  Interpreter? _interpreter;
  List<String> _labels = const ['Cotton', 'Polyester', 'Silk'];

  bool get demoMode => _interpreter == null;
  List<String> get labels => _labels;

  /// Call once at startup. Never throws — missing model just means demo mode.
  Future<void> load() async {
    try {
      _labels = (await rootBundle.loadString(_labelsAsset))
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();
    } catch (_) {/* keep defaults */}

    try {
      _interpreter = await Interpreter.fromAsset(_modelAsset);
    } catch (_) {
      _interpreter = null; // -> demo mode
    }
  }

  /// Run inference on raw image bytes (from camera/gallery).
  /// Returns predictions sorted high -> low confidence.
  Future<List<Prediction>> classify(Uint8List bytes) async {
    if (_interpreter == null) return _demoPredictions();

    final decoded = img.decodeImage(bytes);
    if (decoded == null) return _demoPredictions();

    final input = _preprocess(decoded);
    // Output shape [1, numClasses]; model already ends in softmax.
    final output =
        List.generate(1, (_) => List.filled(_labels.length, 0.0));

    _interpreter!.run(input, output);

    final scores = output.first;
    final preds = <Prediction>[
      for (var i = 0; i < _labels.length; i++)
        Prediction(_labels[i], scores[i].toDouble()),
    ]..sort((a, b) => b.confidence.compareTo(a.confidence));
    return preds;
  }

  /// Resize to 224x224 and lay out as [1,224,224,3] float pixels in 0..255.
  ///
  /// NOTE: the training script bakes MobileNetV2 `preprocess_input`
  /// (the scale-to-[-1,1] step) INSIDE the model graph, so we feed raw
  /// 0..255 values here. If you move normalization out of the model,
  /// normalize here instead.
  List<List<List<List<double>>>> _preprocess(img.Image image) {
    final resized =
        img.copyResize(image, width: _inputSize, height: _inputSize);
    return [
      List.generate(_inputSize, (y) {
        return List.generate(_inputSize, (x) {
          final p = resized.getPixel(x, y);
          return [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()];
        });
      }),
    ];
  }

  /// Random-ish but plausible output so the UI is testable without a model.
  List<Prediction> _demoPredictions() {
    final rng = Random();
    final raw = List.generate(_labels.length, (_) => rng.nextDouble());
    final sum = raw.reduce((a, b) => a + b);
    final preds = <Prediction>[
      for (var i = 0; i < _labels.length; i++)
        Prediction(_labels[i], raw[i] / sum),
    ]..sort((a, b) => b.confidence.compareTo(a.confidence));
    return preds;
  }

  void dispose() => _interpreter?.close();
}
