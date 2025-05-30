import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import AdamW

# —— 修改点：插入自定义 LSTM 层 ——  
class MyLSTMLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_i')
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_i')
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')

        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_f')
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_f')
        self.b_f = self.add_weight(shape=(self.units,), initializer='ones', name='b_f')  # bias_f 通常初始化为 1

        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_c')
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_c')
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', name='b_c')

        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_o')
        self.U_o = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_o')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: [batch, time, features]
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]

        h = tf.zeros((batch_size, self.units))
        c = tf.zeros((batch_size, self.units))

        for t in range(time_steps):
            x_t = inputs[:, t, :]
            i = tf.sigmoid(tf.matmul(x_t, self.W_i) + tf.matmul(h, self.U_i) + self.b_i)
            f = tf.sigmoid(tf.matmul(x_t, self.W_f) + tf.matmul(h, self.U_f) + self.b_f)
            o = tf.sigmoid(tf.matmul(x_t, self.W_o) + tf.matmul(h, self.U_o) + self.b_o)
            c_hat = tf.tanh(tf.matmul(x_t, self.W_c) + tf.matmul(h, self.U_c) + self.b_c)

            c = f * c + i * c_hat
            h = o * tf.tanh(c)

        return h


def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = MyLSTMLayer(units=32)(inputs)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    optimizer = AdamW(learning_rate=1e-4, weight_decay=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model
