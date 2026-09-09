def build_legacy_dann(timesteps: int, channels: int, lambda_d: float = 1.0, lr: float = 1e-4, domain_loss_weight: float = 0.5):
    """DANN exactly matching the original cross-domain notebook architecture.

    IMPORTANT: the notebook's training code supplies classification labels for BOTH
    source and target samples. This is therefore supervised target training plus a
    domain-adversarial head, not source-labelled/target-unlabelled DANN.
    """
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, SpatialDropout1D, Bidirectional, LSTM, MultiHeadAttention, Add, LayerNormalization, GlobalAveragePooling1D, Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam

    class GradientReversal(tf.keras.layers.Layer):
        def __init__(self, lambda_=1.0, **kwargs):
            super().__init__(**kwargs); self.lambda_ = lambda_
        def call(self, x):
            lambda_ = self.lambda_
            @tf.custom_gradient
            def _reverse(z):
                def grad(dy): return -lambda_ * dy
                return z, grad
            return _reverse(x)

    inp = Input(shape=(timesteps, channels), name="inputs_dann")
    x = Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(inp)
    x = BatchNormalization()(x); x = SpatialDropout1D(0.2)(x); x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(256, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x); x = SpatialDropout1D(0.2)(x); x = MaxPooling1D(2, padding="same")(x)
    x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm_dann")(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=64, name="mha_dann")(x, x)
    x = Add()([x, attn_out]); x = LayerNormalization()(x)
    features = GlobalAveragePooling1D(name="gap_dann")(x)

    h_cls = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(features); h_cls = Dropout(0.4)(h_cls)
    out_cls = Dense(2, activation="softmax", name="cls_output")(h_cls)
    grl = GradientReversal(lambda_=lambda_d, name="grad_reverse")(features)
    h_dom = Dense(64, activation="relu")(grl)
    out_dom = Dense(2, activation="softmax", name="dom_output")(h_dom)

    model = Model(inp, [out_cls, out_dom], name="DANN_IDS")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"cls_output":"categorical_crossentropy", "dom_output":"categorical_crossentropy"},
        loss_weights={"cls_output":1.0, "dom_output":domain_loss_weight},
        metrics={"cls_output":["accuracy"], "dom_output":["accuracy"]},
    )
    return model
