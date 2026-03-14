"""
Fingerprint to Blood Group Prediction Model
============================================
Dataset Structure Expected:
    dataset/
        A+/  (folder with fingerprint images for blood group A+)
        A-/
        B+/
        B-/
        AB+/
        AB-/
        O+/
        O-/

Usage:
    python train_model.py --dataset ../dataset --epochs 30 --output ../model_output
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.applications import MobileNetV2

# ── TensorFlow / Keras ──────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU enabled: {gpus}")

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
BLOOD_GROUPS  = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# ────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """Transfer-learning model built on MobileNetV2."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False          # freeze base first

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def get_generators(dataset_path: str):
    """Return train / val / test generators."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )
    test_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        color_mode="rgb",
    )
    val_gen = train_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        color_mode="rgb",
    )
    return train_gen, val_gen


def plot_history(history, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
    plt.close()
    print("✅  Saved training_history.png")


def plot_confusion(model, val_gen, class_names, output_dir: str):
    val_gen.reset()
    preds = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[: len(y_pred)]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("✅  Saved confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def train(dataset_path: str, epochs: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_gen, val_gen = get_generators(dataset_path)
    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    # Save class index mapping
    with open(os.path.join(output_dir, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"✅  Found {num_classes} classes: {class_names}")

    model = build_model(num_classes)
    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    # Phase 1 — frozen base
    print("\n🔥  Phase 1: Training head layers …")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cb_list,
    )

    # Phase 2 — fine-tune top 40 layers of base
    print("\n🔥  Phase 2: Fine-tuning …")
    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=max(epochs // 2, 10),
        callbacks=cb_list,
    )

    # Merge histories for plotting
    combined = {}
    for k in history.history:
        combined[k] = history.history[k] + history2.history[k]
    history.history = combined

    plot_history(history, output_dir)
    plot_confusion(model, val_gen, class_names, output_dir)

    # Save final model in both formats
    model.save(os.path.join(output_dir, "blood_group_model.keras"))
    model.save(os.path.join(output_dir, "blood_group_model.h5"))

    # Save TF-SavedModel for serving
    tf.saved_model.save(model, os.path.join(output_dir, "saved_model"))

    print(f"\n✅  Model saved to: {output_dir}")
    print("    Files created:")
    for f in os.listdir(output_dir):
        print(f"      • {f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Blood Group Fingerprint Model")
    parser.add_argument("--dataset", required=True, help="Path to dataset root folder")
    parser.add_argument("--epochs",  type=int, default=30, help="Number of training epochs")
    parser.add_argument("--output",  default="../model_output", help="Directory to save model & plots")
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.output)