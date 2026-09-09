from __future__ import annotations


def _gradient_reversal_layer(lambda_d: float):
    import tensorflow as tf

    class GradientReversal(tf.keras.layers.Layer):
        def __init__(self, lambda_: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.lambda_ = float(lambda_)

        def call(self, x):
            lambda_ = self.lambda_

            @tf.custom_gradient
            def _reverse(z):
                def grad(dy):
                    return -lambda_ * dy
                return z, grad

            return _reverse(x)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"lambda_": self.lambda_})
            return cfg

    return GradientReversal(lambda_=lambda_d, name="grad_reverse")


def build_mlp_binary(
    input_dim: int,
    *,
    lr: float = 1e-3,
    hidden_1: int = 64,
    hidden_2: int = 32,
    dropout: float = 0.25,
):
    """v0.4.5 historical canonical-tabular control.

    Kept for reproducibility of v0.4.5/v0.4.6 diagnostic runs.  New DANN fairness
    experiments should use ``build_mlp_fair_binary`` so that source-only and DANN
    share the exact encoder + task-head architecture.
    """
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    inp = Input(shape=(int(input_dim),), name="canonical_features")
    x = Dense(hidden_1, activation="relu", kernel_regularizer=l2(1e-4), name="mlp_dense_1")(inp)
    x = BatchNormalization(name="mlp_bn_1")(x)
    x = Dropout(dropout, name="mlp_dropout_1")(x)
    features = Dense(hidden_2, activation="relu", kernel_regularizer=l2(1e-4), name="mlp_embedding")(x)
    out = Dense(2, activation="softmax", name="cls_output")(features)
    model = Model(inp, out, name="MLP_Canonical_Control")
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp_dann(
    input_dim: int,
    *,
    lambda_d: float = 0.1,
    lr: float = 1e-4,
    domain_loss_weight: float = 0.5,
    hidden_1: int = 64,
    hidden_2: int = 32,
    dropout: float = 0.25,
):
    """v0.4.5 historical MLP-DANN control, retained for reproducibility."""
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    inp = Input(shape=(int(input_dim),), name="canonical_features_dann")
    x = Dense(hidden_1, activation="relu", kernel_regularizer=l2(1e-4), name="mlp_dense_1")(inp)
    x = BatchNormalization(name="mlp_bn_1")(x)
    x = Dropout(dropout, name="mlp_dropout_1")(x)
    features = Dense(hidden_2, activation="relu", kernel_regularizer=l2(1e-4), name="mlp_embedding")(x)

    h_cls = Dense(hidden_2, activation="relu", kernel_regularizer=l2(1e-4), name="cls_hidden")(features)
    h_cls = Dropout(dropout, name="cls_dropout")(h_cls)
    out_cls = Dense(2, activation="softmax", name="cls_output")(h_cls)

    grl = _gradient_reversal_layer(lambda_d)(features)
    h_dom = Dense(hidden_2, activation="relu", name="dom_hidden")(grl)
    out_dom = Dense(2, activation="softmax", name="dom_output")(h_dom)

    model = Model(inp, [out_cls, out_dom], name="MLP_DANN_Canonical_Control")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"cls_output": "categorical_crossentropy", "dom_output": "categorical_crossentropy"},
        loss_weights={"cls_output": 1.0, "dom_output": float(domain_loss_weight)},
        metrics={"cls_output": ["accuracy"], "dom_output": ["accuracy"]},
    )
    return model


def _fair_mlp_trunk(inp, hidden_1: int, hidden_2: int, dropout: float):
    """Shared v0.4.7 tabular trunk.

    LayerNorm is intentionally used instead of BatchNorm.  DANN sees source and
    target batches from different domains, and BatchNorm moving statistics can
    become an additional adaptation/confounding mechanism.  LayerNorm is
    sample-local and therefore keeps the source-only vs DANN architecture control
    cleaner.
    """
    from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
    from tensorflow.keras.regularizers import l2

    x = Dense(hidden_1, activation="relu", kernel_regularizer=l2(1e-4), name="fair_dense_1")(inp)
    x = LayerNormalization(name="fair_ln_1")(x)
    x = Dropout(dropout, name="fair_dropout_1")(x)
    features = Dense(hidden_2, activation="relu", kernel_regularizer=l2(1e-4), name="fair_embedding")(x)
    return features


def _fair_task_head(features, hidden_2: int, dropout: float):
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.regularizers import l2

    h = Dense(hidden_2, activation="relu", kernel_regularizer=l2(1e-4), name="fair_cls_hidden")(features)
    h = Dropout(dropout, name="fair_cls_dropout")(h)
    return Dense(2, activation="softmax", name="cls_output")(h)


def build_mlp_fair_binary(
    input_dim: int,
    *,
    lr: float = 1e-3,
    hidden_1: int = 64,
    hidden_2: int = 32,
    dropout: float = 0.25,
):
    """v0.4.7 apples-to-apples source-only tabular control.

    The encoder and task head are exactly the same as in ``build_mlp_fair_dann``;
    the latter only adds a GRL + domain head.  This removes the architecture
    mismatch present in v0.4.5/v0.4.6 diagnostics.
    """
    from tensorflow.keras import Input, Model
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(int(input_dim),), name="fair_canonical_features")
    features = _fair_mlp_trunk(inp, hidden_1, hidden_2, dropout)
    out_cls = _fair_task_head(features, hidden_2, dropout)
    model = Model(inp, out_cls, name="MLP_Fair_Canonical_Control")
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp_fair_dann(
    input_dim: int,
    *,
    lambda_d: float = 0.1,
    lr: float = 1e-3,
    domain_loss_weight: float = 0.5,
    hidden_1: int = 64,
    hidden_2: int = 32,
    dropout: float = 0.25,
):
    """v0.4.7 apples-to-apples MLP + DANN control.

    Same encoder and task head as ``build_mlp_fair_binary``.  The only structural
    difference is the domain branch attached through gradient reversal.
    """
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(int(input_dim),), name="fair_canonical_features")
    features = _fair_mlp_trunk(inp, hidden_1, hidden_2, dropout)
    out_cls = _fair_task_head(features, hidden_2, dropout)

    grl = _gradient_reversal_layer(lambda_d)(features)
    h_dom = Dense(hidden_2, activation="relu", name="dom_hidden")(grl)
    out_dom = Dense(2, activation="softmax", name="dom_output")(h_dom)

    model = Model(inp, [out_cls, out_dom], name="MLP_Fair_DANN_Canonical_Control")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"cls_output": "categorical_crossentropy", "dom_output": "categorical_crossentropy"},
        loss_weights={"cls_output": 1.0, "dom_output": float(domain_loss_weight)},
        metrics={"cls_output": ["accuracy"], "dom_output": ["accuracy"]},
    )
    return model


# ---------------------------------------------------------------------------
# v0.4.8 staged BatchNorm-fair control
# ---------------------------------------------------------------------------

def _bnfair_mlp_trunk(inp, hidden_1: int, hidden_2: int, dropout: float):
    """Trunk matching the successful v0.4.5 MLP control.

    BatchNorm is intentionally retained because the source-only v0.4.5 control
    transferred well to CICIoT2023.  During DANN adaptation v0.4.8 freezes this
    BatchNorm layer after source pretraining, so unlabeled target batches cannot
    overwrite source normalization statistics and confound the adversarial test.
    """
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.regularizers import l2

    x = Dense(hidden_1, activation="relu", kernel_regularizer=l2(1e-4), name="bnfair_dense_1")(inp)
    x = BatchNormalization(name="bnfair_bn_1")(x)
    x = Dropout(dropout, name="bnfair_dropout_1")(x)
    features = Dense(hidden_2, activation="relu", kernel_regularizer=l2(1e-4), name="bnfair_embedding")(x)
    return features


def build_mlp_bn_fair_binary(
    input_dim: int,
    *,
    lr: float = 1e-3,
    hidden_1: int = 64,
    hidden_2: int = 32,
    dropout: float = 0.25,
):
    """Source-pretraining model for the v0.4.8 staged DANN fairness control."""
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(int(input_dim),), name="bnfair_canonical_features")
    features = _bnfair_mlp_trunk(inp, hidden_1, hidden_2, dropout)
    out_cls = Dense(2, activation="softmax", name="cls_output")(features)
    model = Model(inp, out_cls, name="MLP_BN_Fair_Canonical_Control")
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp_bn_fair_dann(
    input_dim: int,
    *,
    lambda_d: float = 0.1,
    lr: float = 1e-3,
    domain_loss_weight: float = 0.5,
    hidden_1: int = 64,
    hidden_2: int = 32,
    dropout: float = 0.25,
):
    """DANN with exactly the same task path as ``build_mlp_bn_fair_binary``.

    The caller should initialize shared layers from a source-pretrained model and
    freeze ``bnfair_bn_1`` before adaptation.  Then lambda=0 is a true
    zero-strength adversarial continuation control: the domain head learns, but
    its gradient cannot reach the shared encoder.
    """
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(int(input_dim),), name="bnfair_canonical_features")
    features = _bnfair_mlp_trunk(inp, hidden_1, hidden_2, dropout)
    out_cls = Dense(2, activation="softmax", name="cls_output")(features)

    grl = _gradient_reversal_layer(lambda_d)(features)
    h_dom = Dense(hidden_2, activation="relu", name="dom_hidden")(grl)
    out_dom = Dense(2, activation="softmax", name="dom_output")(h_dom)

    model = Model(inp, [out_cls, out_dom], name="MLP_BN_Fair_DANN_Canonical_Control")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"cls_output": "categorical_crossentropy", "dom_output": "categorical_crossentropy"},
        loss_weights={"cls_output": 1.0, "dom_output": float(domain_loss_weight)},
        metrics={"cls_output": ["accuracy"], "dom_output": ["accuracy"]},
    )
    return model


def shared_layer_weights(model) -> dict[str, list]:
    """Return weights for task-path layers that can be copied into staged DANN."""
    names = {"bnfair_dense_1", "bnfair_bn_1", "bnfair_embedding", "cls_output"}
    out = {}
    for layer in model.layers:
        if layer.name in names:
            out[layer.name] = layer.get_weights()
    return out


def load_shared_layer_weights(model, weights_by_name: dict[str, list], *, freeze_batchnorm: bool = True):
    """Load source-pretrained task-path weights into a DANN model by layer name."""
    missing = []
    for name, weights in weights_by_name.items():
        try:
            model.get_layer(name).set_weights(weights)
        except ValueError:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing staged shared layers in DANN model: {missing}")
    if freeze_batchnorm:
        model.get_layer("bnfair_bn_1").trainable = False
    return model
