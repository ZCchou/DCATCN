from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import AdamW
def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=1)(x)
    x = layers.Dropout(0.1)(x)
    # BiGRU
    x = layers.Bidirectional(layers.GRU(16, return_sequences=False))(x)
    att = layers.Dense(32, activation='sigmoid')(x)
    x = layers.Multiply()([x, att])
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    optimizer = AdamW(learning_rate=1e-4, weight_decay=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model