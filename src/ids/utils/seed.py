import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Best-effort deterministic seed setup.

    PYTHONHASHSEED only fully affects set/dict hashing when set before Python starts.
    For strict cross-domain legacy reproduction, launch with PYTHONHASHSEED=... .
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass
