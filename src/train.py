import argparse
import os

import tensorflow as tf
from matplotlib import pyplot as plt

from models import DogResNet50


def main(args):
    print(args)
    train_path = os.path.join(args.filepath, 'train')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size)

    class_names = [class_name[10:].lower().replace('-', '_') for class_name in train_ds.class_names]
    
    # Train data
    model = DogResNet50(inputs=(224, 224, 3), n_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(train_ds,
                        epochs=args.epochs,
                        validation_data=val_ds)   

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dog breed recognition train')
    parser.add_argument('filepath', type=str, help='Path to dataset.')
    parser.add_argument('output', type=str, help='Path to save the model.')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size.', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs.', default=100)
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate', type=float, help='Learning rate.', default=1e-3)

    args = parser.parse_args()

    main(args)
