from __future__ import annotations

import numpy as np


def _classification_output(out):
    return out[0] if isinstance(out, (list, tuple)) else out


def fgsm(
    model,
    x,
    y,
    epsilon: float,
    clip_min: float | None = None,
    clip_max: float | None = None,
    *,
    project_fn=None,
):
    """FGSM in model-input space for a Keras classifier.

    ``project_fn`` is mandatory for final E6 evidence; without it this helper is
    only an unconstrained diagnostic.
    """
    import tensorflow as tf

    x_tf = tf.convert_to_tensor(np.asarray(x), dtype=tf.float32)
    y_tf = tf.convert_to_tensor(np.asarray(y))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        out = model(x_tf, training=False)
        probs = _classification_output(out)
        loss = loss_fn(y_tf, probs)
    grad = tape.gradient(loss, x_tf)
    adv = x_tf + epsilon * tf.sign(grad)
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else clip_min
        hi = np.inf if clip_max is None else clip_max
        adv = tf.clip_by_value(adv, lo, hi)
    result = adv.numpy()
    if project_fn is not None:
        result = project_fn(result, np.asarray(x, dtype=np.float32), float(epsilon))
    return np.asarray(result, dtype=np.float32)


def pgd(
    model, x, y, epsilon: float, alpha: float, steps: int,
    clip_min: float | None = None, clip_max: float | None = None,
    *, project_fn=None, random_start: bool = False, seed: int | None = None,
):
    """Projected gradient descent in model-input space.

    When supplied, ``project_fn`` is applied after the random start and after
    every gradient step, so the final E6 runner never emits an infeasible point.
    """
    import tensorflow as tf

    x0 = tf.convert_to_tensor(np.asarray(x), dtype=tf.float32)
    y_tf = tf.convert_to_tensor(np.asarray(y))
    if random_start and float(epsilon) > 0:
        rng = np.random.default_rng(seed)
        start = np.asarray(x, dtype=np.float32) + rng.uniform(
            -float(epsilon), float(epsilon), size=np.shape(x)
        ).astype(np.float32)
        if project_fn is not None:
            start = project_fn(start, np.asarray(x, dtype=np.float32), float(epsilon))
        adv = tf.convert_to_tensor(start, dtype=tf.float32)
    else:
        adv = tf.identity(x0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            out = model(adv, training=False)
            probs = _classification_output(out)
            loss = loss_fn(y_tf, probs)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.minimum(tf.maximum(adv, x0 - epsilon), x0 + epsilon)
        if clip_min is not None or clip_max is not None:
            lo = -np.inf if clip_min is None else clip_min
            hi = np.inf if clip_max is None else clip_max
            adv = tf.clip_by_value(adv, lo, hi)
        if project_fn is not None:
            projected = project_fn(adv.numpy(), x0.numpy(), float(epsilon))
            adv = tf.convert_to_tensor(projected, dtype=tf.float32)
    return adv.numpy().astype(np.float32)
