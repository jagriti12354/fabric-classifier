import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'classifier.dart';

void main() => runApp(const FabricApp());

class FabricApp extends StatelessWidget {
  const FabricApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fiber ID',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF4E6E58), // muted textile green
        brightness: Brightness.light,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _classifier = FiberClassifier();
  final _picker = ImagePicker();

  File? _image;
  List<Prediction> _results = const [];
  bool _busy = false;
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _classifier.load().then((_) => setState(() => _ready = true));
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  Future<void> _pick(ImageSource source) async {
    final picked = await _picker.pickImage(source: source, maxWidth: 1024);
    if (picked == null) return;

    setState(() {
      _image = File(picked.path);
      _results = const [];
      _busy = true;
    });

    final Uint8List bytes = await picked.readAsBytes();
    final results = await _classifier.classify(bytes);

    if (!mounted) return;
    setState(() {
      _results = results;
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Fiber ID'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_ready && _classifier.demoMode) _demoBanner(scheme),
              Expanded(child: _preview(scheme)),
              const SizedBox(height: 16),
              _resultsPanel(scheme),
              const SizedBox(height: 16),
              _buttons(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _demoBanner(ColorScheme scheme) => Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: scheme.tertiaryContainer,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(
          children: [
            const Icon(Icons.info_outline, size: 18),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                'Demo mode — no model bundled yet. Results are random.',
                style: TextStyle(color: scheme.onTertiaryContainer),
              ),
            ),
          ],
        ),
      );

  Widget _preview(ColorScheme scheme) {
    return Container(
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(16),
      ),
      clipBehavior: Clip.antiAlias,
      child: _image == null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.texture, size: 64, color: scheme.outline),
                  const SizedBox(height: 12),
                  Text('Take a close-up of the fabric weave',
                      style: TextStyle(color: scheme.outline)),
                ],
              ),
            )
          : Image.file(_image!, fit: BoxFit.cover),
    );
  }

  Widget _resultsPanel(ColorScheme scheme) {
    if (_busy) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 24),
        child: Center(child: CircularProgressIndicator()),
      );
    }
    if (_results.isEmpty) return const SizedBox.shrink();

    return Column(
      children: [
        for (var i = 0; i < _results.length; i++)
          _bar(_results[i], highlight: i == 0, scheme: scheme),
      ],
    );
  }

  Widget _bar(Prediction p,
      {required bool highlight, required ColorScheme scheme}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                p.label[0].toUpperCase() + p.label.substring(1),
                style: TextStyle(
                  fontWeight: highlight ? FontWeight.w700 : FontWeight.w400,
                ),
              ),
              Text('${(p.confidence * 100).toStringAsFixed(1)}%'),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: p.confidence,
              minHeight: 8,
              backgroundColor: scheme.surfaceContainerHighest,
              color: highlight ? scheme.primary : scheme.outlineVariant,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buttons() {
    return Row(
      children: [
        Expanded(
          child: FilledButton.icon(
            onPressed: _busy ? null : () => _pick(ImageSource.camera),
            icon: const Icon(Icons.photo_camera_outlined),
            label: const Text('Camera'),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: FilledButton.tonalIcon(
            onPressed: _busy ? null : () => _pick(ImageSource.gallery),
            icon: const Icon(Icons.photo_library_outlined),
            label: const Text('Gallery'),
          ),
        ),
      ],
    );
  }
}
