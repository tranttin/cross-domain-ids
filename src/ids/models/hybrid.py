def build_legacy_multiclass_hybrid(timesteps: int, channels: int, num_classes: int, lr: float = 1e-4):
    """Exact architecture from cse-cic-ids2018-cnn-lstm-attention-hybrid.ipynb.

    Note: this differs from the architecture/hyperparameters described in the paper.
    """
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Bidirectional, LSTM, MultiHeadAttention, GlobalAveragePooling1D, Dense
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=(timesteps, channels), name="inputs")
    x = Conv1D(128, 3, activation="relu", padding="same", kernel_regularizer=l2(1e-3))(inputs)
    x = BatchNormalization()(x); x = MaxPooling1D(2, padding="same")(x); x = Dropout(0.3)(x)
    x = Conv1D(256, 3, activation="relu", padding="same", kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x); x = MaxPooling1D(2, padding="same")(x); x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm")(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=64, name="mha")(x, x)
    x = x + attn_out
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", name="softmax")(x)
    model = Model(inputs, outputs, name="CNN_BiLSTM_Attn")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="Precision"), tf.keras.metrics.Recall(name="Recall")],
    )
    return model


def build_legacy_binary_hybrid(timesteps: int, channels: int, lr: float = 1e-3):
    """Exact binary baseline architecture from cross-domain.ipynb."""
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, SpatialDropout1D, Bidirectional, LSTM, MultiHeadAttention, Add, LayerNormalization, GlobalAveragePooling1D, Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(timesteps, channels), name="inputs")
    x = Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(inp)
    x = BatchNormalization()(x); x = SpatialDropout1D(0.2)(x); x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(256, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x); x = SpatialDropout1D(0.2)(x); x = MaxPooling1D(2, padding="same")(x)
    x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm")(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=64, name="mha")(x, x)
    x = Add()([x, attn_out]); x = LayerNormalization()(x)
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x); x = Dropout(0.4)(x)
    out = Dense(2, activation="softmax", name="cls_output")(x)
    model = Model(inp, out, name="Baseline_IDS")
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
