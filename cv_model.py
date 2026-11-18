"""Age-regression exercise using tf.data and current Keras APIs.

Historical 0..1 pixel scaling is preserved. ImageNet weights require explicit
opt-in; default construction and smoke tests never download weights.
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf


def load_images(path, batch_size=16, training=True, split='train', image_size=(224, 224)):
    path = Path(path)
    labels = pd.read_csv(path / 'labels.csv')
    if labels.empty or labels[['file_name', 'real_age']].isna().any().any():
        raise ValueError('Nonempty labels with file_name and real_age are required')
    paths = [(path / 'final_files' / str(name)).resolve() for name in labels.file_name]
    base = (path / 'final_files').resolve()
    if any(not name.is_relative_to(base) or not name.is_file() for name in paths):
        raise ValueError('Every image must exist inside final_files')
    boundary = int(len(labels) * .25)
    selection = slice(boundary, None) if split == 'train' else slice(None, boundary) if split == 'validation' else slice(None)
    paths = [str(p) for p in paths[selection]]
    ages = labels.real_age.to_numpy(dtype='float32')[selection]
    if not paths:
        raise ValueError('Not enough rows for the requested split')
    dataset = tf.data.Dataset.from_tensor_slices((paths, ages))
    if training:
        dataset = dataset.shuffle(len(paths), seed=12345)

    def decode(filename, age):
        image = tf.io.decode_image(tf.io.read_file(filename), channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size) / 255.0
        if training:
            image = tf.image.random_flip_left_right(image, seed=12345)
        return image, age

    return dataset.map(decode, num_parallel_calls=1).batch(batch_size).prefetch(1)


def create_model(input_shape=(224, 224, 3), weights=None):
    tf.keras.utils.set_random_seed(12345)
    backbone = tf.keras.applications.ResNet50(input_shape=input_shape, weights=weights, include_top=False)
    model = tf.keras.Sequential([backbone, tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(100, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dense(1, activation='relu')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001), loss='mse', metrics=['mae'])
    return model


def train_model(model, train_data, test_data, epochs=5, steps_per_epoch=None, validation_steps=None):
    model.fit(train_data, validation_data=test_data, epochs=epochs,
              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, verbose=2)
    return model
