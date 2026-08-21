"""Train the fabric fiber classifier and export a TFLite model for the Flutter app.

Usage:
    python train_fiber_classifier.py

Expects a Roboflow-style folder-classification export at ./dataset with
train/, valid/, (optional) test/ subfolders, each containing one folder per
class. Produces:
    fiber_classifier.tflite  - drop into assets/
    labels.txt                - drop into assets/ (class order matches the model)
"""

import pathlib

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_DIR = pathlib.Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 6
FINE_TUNE_AT = 100  # unfreeze MobileNetV2 layers from this index onward


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names

    test_ds = None
    if TEST_DIR.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    valid_ds = valid_ds.cache().prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, valid_ds, test_ds, class_names


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ]
    )

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    # preprocess_input scales 0..255 -> -1..1; baked into the graph so the
    # Flutter app can feed raw pixel values straight from the decoded image.
    x = tf.keras.layers.Lambda(preprocess_input, name="preprocess")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def main():
    train_ds, valid_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model, base = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n== Stage 1: training classifier head (base frozen) ==")
    model.fit(train_ds, validation_data=valid_ds, epochs=HEAD_EPOCHS)

    print("\n== Stage 2: fine-tuning top MobileNetV2 layers ==")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=valid_ds, epochs=FINE_TUNE_EPOCHS)

    if test_ds is not None:
        loss, acc = model.evaluate(test_ds)
        print(f"\nTest accuracy: {acc:.4f} (loss {loss:.4f})")

    labels_path = pathlib.Path("labels.txt")
    labels_path.write_text("\n".join(class_names) + "\n")
    print(f"Wrote {labels_path} ({num_classes} classes)")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = pathlib.Path("fiber_classifier.tflite")
    tflite_path.write_bytes(tflite_model)
    print(f"Wrote {tflite_path} ({len(tflite_model) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
