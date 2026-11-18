import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip('tensorflow')
from cv_model import create_model, load_images  # noqa: E402


def test_generated_images_and_model_roundtrip(tmp_path):
    directory = tmp_path / 'final_files'
    directory.mkdir()
    for i in range(8):
        tf.io.write_file(str(directory / f'{i}.png'), tf.io.encode_png(tf.ones((32, 32, 3), tf.uint8) * (i * 20)))
    pd.DataFrame({'file_name': [f'{i}.png' for i in range(8)], 'real_age': np.arange(8) + 20}).to_csv(tmp_path / 'labels.csv', index=False)
    train = load_images(tmp_path, batch_size=2, image_size=(32, 32))
    valid = load_images(tmp_path, batch_size=2, training=False, split='validation', image_size=(32, 32))
    assert sum(len(y) for _, y in train) == 6
    assert sum(len(y) for _, y in valid) == 2
    X, y = next(iter(train))
    assert 0 <= float(tf.reduce_min(X)) <= float(tf.reduce_max(X)) <= 1
    model = create_model((32, 32, 3), weights=None)
    # One real gradient update, random weights; no claim about age accuracy.
    loss = model.train_on_batch(X, y)
    assert np.isfinite(loss).all()
    predicted = model(X, training=False).numpy()
    path = tmp_path / 'smoke.keras'
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(predicted, restored(X, training=False).numpy(), rtol=1e-5)
