from keras.optimizers.schedules.learning_rate_schedule import PolynomialDecay
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import AdamW

# ------------------------------
# Attention layer definition: generate softmax attention weights from fully connected layer outputs
# ------------------------------
# ------------------------------ Attention Mechanism ------------------------------
class AttentionLayer(layers.Layer):
    def __init__(self, units=128, num_heads=2, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.query = layers.Dense(units, use_bias=False)
        self.key = layers.Dense(units, use_bias=False)
        self.value = layers.Dense(units, use_bias=False)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads)
        self.projection = layers.Dense(units)

    def call(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)
        attention_output = self.attention(Q, K, V)
        return self.projection(attention_output) + inputs

# ------------------------------ TCN module------------------------------
class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation_rate, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = tfa.layers.WeightNormalization(
            layers.Conv1D(filters=n_outputs,
                          kernel_size=kernel_size,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding='causal',
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        )
        self.relu1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = tfa.layers.WeightNormalization(
            layers.Conv1D(filters=n_outputs,
                          kernel_size=kernel_size,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding='causal',
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        )
        self.relu2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout)

        if n_inputs != n_outputs:
            self.downsample = tfa.layers.WeightNormalization(
                layers.Conv1D(filters=n_outputs,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            )
        else:
            self.downsample = None
        self.final_relu = layers.ReLU()

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out, training=training)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out, training=training)

        res = self.downsample(x) if self.downsample is not None else x
        return self.final_relu(layers.Add()([out, res]))

class TemporalConvNet(tf.keras.Model):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.tcn_layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.tcn_layers.append(
                TemporalBlock(n_inputs=in_channels,
                              n_outputs=out_channels,
                              kernel_size=kernel_size,
                              dilation_rate=dilation_rate,
                              dropout=dropout)
            )

    def call(self, x, training=False):
        out = x
        for layer in self.tcn_layers:
            out = layer(out, training=training)
        return out

class BiTCNBlock(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation_rate, dropout=0.05, **kwargs):
        """
        :param n_inputs: Number of input channels
        :param n_outputs: Number of output channels per branch
        :param kernel_size: Convolution kernel size
        :param dilation_rate: Dilation rate
        :param dropout: Dropout probability
        """
        super(BiTCNBlock, self).__init__(**kwargs)
        self.forward_block = TemporalBlock(n_inputs=n_inputs,
                                           n_outputs=n_outputs,
                                           kernel_size=kernel_size,
                                           dilation_rate=dilation_rate,
                                           dropout=dropout)
        self.backward_block = TemporalBlock(n_inputs=n_inputs,
                                            n_outputs=n_outputs,
                                            kernel_size=kernel_size,
                                            dilation_rate=dilation_rate,
                                            dropout=dropout)
        self.skip_conv = layers.Conv1D(2 * n_outputs, 1, padding='same')
        self.final_relu = layers.ReLU()

    def call(self, inputs, training=False):
        # Forward branch
        f = self.forward_block(inputs, training=training)
        # Backward branch: reverse along time dimension first
        r = tf.reverse(inputs, axis=[1])
        b = self.backward_block(r, training=training)
        # Reverse back
        b = tf.reverse(b, axis=[1])
        # Concatenate two branches
        merged = tf.concat([f, b], axis=-1)
        residual = self.skip_conv(inputs)
        return self.final_relu(layers.Add()([merged, residual]))


class BiTCN(tf.keras.Model):
    def __init__(self, num_inputs, num_layers, filters, kernel_size=3, dropout=0.05, feature_extractor=False, **kwargs):
        """
        :param num_inputs: Dimensionality of input features
        :param num_layers: Number of stacked BiTCNBlock layers
        :param filters: Number of output channels for each TemporalBlock in BiTCNBlock (per direction)
        :param kernel_size: Size of the convolutional kernel
        :param dropout: Dropout probability
        :param feature_extractor: If True, only stack BiTCNBlocks and return a 3D feature map;
                                  if False, apply GlobalAveragePooling1D and Dense to produce final prediction
        """
        super(BiTCN, self).__init__(**kwargs)
        self.blocks = []
        for i in range(num_layers):
            dilation_rate = 2 ** i
            if i == 0:
                in_channels = num_inputs
            else:
                in_channels = 2 * filters  # Output of the previous layer is 2 * filters
            self.blocks.append(
                BiTCNBlock(n_inputs=in_channels,
                           n_outputs=filters,
                           kernel_size=kernel_size,
                           dilation_rate=dilation_rate,
                           dropout=dropout)
            )
        self.feature_extractor = feature_extractor
        if not self.feature_extractor:
            self.global_pool = layers.GlobalAveragePooling1D()
            self.dense = layers.Dense(1)


    def call(self, x, training=False):
        out = x
        for block in self.blocks:
            out = block(out, training=training)
        if self.feature_extractor:
            return out
        else:
            out = self.global_pool(out)
            return self.dense(out)


def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Forward TCN: 2 layers, channels=16
    forward_tcn = TemporalConvNet(
        num_inputs=input_shape[1],
        num_channels=[16, 16],
        kernel_size=3,
        dropout=0.05
    )
    f = forward_tcn(inputs)  # (batch, T, 16)

    # Backward TCN: 2 layers, channels=16
    r = tf.reverse(inputs, axis=[1])
    backward_tcn = TemporalConvNet(
        num_inputs=input_shape[1],
        num_channels=[16, 16],
        kernel_size=3,
        dropout=0.05
    )
    b_rev = backward_tcn(r)
    b = tf.reverse(b_rev, axis=[1])  # (batch, T, 16)

    # Single-head cross attention
    att = layers.MultiHeadAttention(num_heads=1, key_dim=16)(
        query=f, value=b, key=b
    )  # (batch, T, 16)

    # Feature concatenation
    x_cat = layers.Concatenate()([f, att])  # (batch, T, 32)

    # Pooling & output
    x_pool = layers.GlobalAveragePooling1D()(x_cat)  # (batch, 32)
    outputs = layers.Dense(1)(x_pool)

    # Compile
    lr_schedule = PolynomialDecay(1e-4, decay_steps=10000, end_learning_rate=1e-6)
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=0.001)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss='mse')
    return model
