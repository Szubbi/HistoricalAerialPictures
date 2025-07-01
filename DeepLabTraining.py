import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from DeepLabModel import DeeplabV3Plus
from keras import backend as K

# Creating Metrics for instance segmentation
# Dice Coefficient
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(tf.argmax(y_pred, axis=-1), tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Mean IoU
class MeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# Dataset functions 
def load_image_label(image_path, label_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [IMAGE_SIZE, IMAGE_SIZE], method='nearest')
    label = tf.cast(label, tf.float32)

    return image, label

def create_dataset(images_dir, labels_dir):
    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')])
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.png')])

    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
    dataset = dataset.map(load_image_label, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(100).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

# Display sample 
def display_samples(dataset, num_samples=8):
    images, labels = [], []
    for image_batch, label_batch in dataset.unbatch().shuffle(1000).take(num_samples):
        images.append(image_batch.numpy().squeeze())
        labels.append(label_batch.numpy().squeeze())

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_samples):
        row = i // 2
        col = (i % 2) * 2
        axes[row, col].imshow(images[i], cmap='gray')
        axes[row, col].set_title("Image")
        axes[row, col].axis('off')

        axes[row, col + 1].imshow(labels[i], cmap='gray')
        axes[row, col + 1].set_title("Label")
        axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    IMAGE_SIZE = 640
    NUM_CLASSES = 1
    BATCH_SIZE = 8
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = create_dataset("dataset/images/train", "dataset/labels/train")
    val_dataset = create_dataset("dataset/images/val", "dataset/labels/val")
    test_dataset = create_dataset("dataset/images/test", "dataset/labels/test")

    # Creating coolback funtions
    # Early stopping: stop training if val_loss doesn't improve for 5 epochs
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True)

    # Learning rate decay: reduce LR by half if val_loss doesn't improve for 3 epochs
    lr_decay = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1)

    model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            'accuracy',
            dice_coef,
            MeanIoU(num_classes=NUM_CLASSES)])
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stop, lr_decay]
    )
