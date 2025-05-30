from keras.optimizers.schedules.learning_rate_schedule import PolynomialDecay
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import AdamW

# ------------------------------ TCN ------------------------------
class TemporalBlock(layers.Layer):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation_rate, dropout=0.2):
        super().__init__()
        self.sep1 = layers.SeparableConv1D(n_outputs, kernel_size,
                                           dilation_rate=dilation_rate, padding='causal')
        self.ln1  = layers.LayerNormalization()
        self.act1 = layers.Activation('gelu')
        self.drop1= layers.Dropout(dropout)
        self.sep2 = layers.SeparableConv1D(n_outputs, kernel_size,dilation_rate=dilation_rate, padding='causal')
        self.ln2  = layers.LayerNormalization()
        self.act2 = layers.Activation('gelu')
        self.drop2= layers.Dropout(dropout)
        self.down = layers.Conv1D(n_outputs, kernel_size=1, padding='same')
        self.final_ln = layers.LayerNormalization()
        self.final_act= layers.Activation('gelu')

    def call(self, x, training=False):
        out = self.sep1(x)
        out = self.ln1(out)
        out = self.act1(out)
        out = self.drop1(out, training=training)
        out = self.sep2(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.drop2(out, training=training)
        res = self.down(x)
        out = out + res
        out = self.final_ln(out)
        return self.final_act(out)

class BiTCNBlock(layers.Layer):
    def __init__(self, channels, kernel_size, dilation_rate, dropout=0.05):
        super().__init__()
        self.fwd = TemporalBlock(channels, channels, kernel_size, dilation_rate, dropout)
        self.bwd = TemporalBlock(channels, channels, kernel_size, dilation_rate, dropout)
        self.skip= layers.Conv1D(2*channels, 1, padding='same')
        self.ln  = layers.LayerNormalization()
        self.act = layers.Activation('gelu')

    def call(self, x, training=False):
        f = self.fwd(x, training=training)
        r = tf.reverse(x, axis=[1])
        b = self.bwd(r, training=training)
        b = tf.reverse(b, axis=[1])
        merged = tf.concat([f, b], axis=-1)
        out = merged + self.skip(x)
        out = self.ln(out)
        return self.act(out)

class BiTCN(models.Model):
    def __init__(self, num_inputs, num_layers, channels, kernel_size=3, dropout=0.05):
        super().__init__()
        self.blocks = []
        for i in range(num_layers):
            self.blocks.append(
                BiTCNBlock(channels, kernel_size, dilation_rate=2**i, dropout=dropout)
            )

    def call(self, x, training=False):
        for blk in self.blocks:
            x = blk(x, training=training)
        return x

def multi_scale_sepconv(x, filters, dropout=0.1):
    b1 = layers.SeparableConv1D(filters, 3, padding='same', dilation_rate=1)(x)
    b2 = layers.SeparableConv1D(filters, 3, padding='same', dilation_rate=2)(x)
    b3 = layers.SeparableConv1D(filters, 5, padding='same', dilation_rate=1)(x)
    out = layers.Concatenate()([b1, b2, b3])
    out = layers.LayerNormalization()(out)
    out = layers.Activation('gelu')(out)
    return layers.Dropout(dropout)(out)

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x0 = multi_scale_sepconv(inputs, filters=32, dropout=0.1)  # (batch,T,32)

    x1 = BiTCN(num_inputs=32, num_layers=4, channels=32,
               kernel_size=3, dropout=0.05)(x0)  # (batch,T,64)

    res = layers.Conv1D(64,1,padding='same')(x0)
    x2 = layers.Add()([x1, res])
    x2 = layers.LayerNormalization()(x2)
    att = layers.Attention(use_scale=True)([x2, x2])
    x3 = layers.Add()([x2, att])
    ff = layers.Dense(128, activation='gelu')(x3)
    ff = layers.Dropout(0.1)(ff)
    ff = layers.Dense(64)(ff)
    ff = layers.Dropout(0.1)(ff)
    x3 = layers.Add()([x3, ff])
    x3 = layers.LayerNormalization()(x3)
    x4 = layers.GlobalAveragePooling1D()(x3)
    x4 = layers.Dropout(0.2)(x4)
    outputs = layers.Dense(1)(x4)

    lr_sched = PolynomialDecay(1e-4, decay_steps=10000, end_learning_rate=1e-6)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=AdamW(learning_rate=lr_sched, weight_decay=0.001), loss='mse')
    return model
