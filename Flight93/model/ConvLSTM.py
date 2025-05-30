from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import AdamW
# 4. ConvLSTM（使用 ConvLSTM2D 处理 1D 序列）
def build_model(input_shape):
    """
    input_shape = (time_steps, features)
    """
    inputs = layers.Input(shape=input_shape)
    # 变成 5D tensor: (batch, time, rows=features, cols=1, channels=1)
    x = layers.Reshape((input_shape[0], input_shape[1], 1, 1))(inputs)

    # ConvLSTM2D expects 5D: (batch, time, rows, cols, channels)
    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(3, 1),
        padding='same',
        return_sequences=False
    )(x)

    x = layers.Flatten()(x)
    att = layers.Dense(x.shape[-1], activation='sigmoid')(x)
    x = layers.Multiply()([x, att])
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    optimizer = AdamW(learning_rate=1e-4, weight_decay=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model