import tensorflow as tf


def DogResNet50(inputs, n_classes):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    features = tf.keras.applications.ResNet50(
        input_shape=inputs,
        include_top=False,
        weights='imagenet'
    )

    inputs = tf.keras.Input(shape=inputs)

    x = normalization_layer(inputs)
    x = data_augmentation(x)
    x = features(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(.2)(x)
    outputs = tf.keras.layers.Dense(n_classes)(x)

    return tf.keras.Model(inputs, outputs)
