# Multi-Channel Attention Layer
class MultiChannelAttentionLayer(Layer):
    def __init__(self, num_channels, **kwargs):
        super(MultiChannelAttentionLayer, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.attention_dense_layers = []

    def build(self, input_shape):
        if input_shape[-1] % self.num_channels != 0:
            raise ValueError(
                f"Input dimension ({input_shape[-1]}) must be divisible by the number of channels ({self.num_channels})"
            )

        channel_dim = input_shape[-1] // self.num_channels
        for _ in range(self.num_channels):
            self.attention_dense_layers.append(
                {
                    "W": self.add_weight(
                        name="attention_weight",
                        shape=(channel_dim, 1),
                        initializer="normal",
                    ),
                    "b": self.add_weight(
                        name="attention_bias",
                        shape=(1,),
                        initializer="zeros",
                    ),
                }
            )
        super(MultiChannelAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        split_channels = tf.split(inputs, num_or_size_splits=self.num_channels, axis=-1)
        attended_channels = []

        for idx, x in enumerate(split_channels):
            W = self.attention_dense_layers[idx]["W"]
            b = self.attention_dense_layers[idx]["b"]
            e = K.tanh(K.dot(x, W) + b)
            a = K.softmax(e, axis=1)
            output = x * a
            attended_channels.append(output)

        return Concatenate(axis=-1)(attended_channels)


# Spatial Attention Layer
class SpatialAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = Conv2D(filters=1, kernel_size=7, padding="same", activation="sigmoid")
        super(SpatialAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        avg_pool = K.mean(inputs, axis=-1, keepdims=True)
        max_pool = K.max(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)
        return Multiply()([inputs, attention])


# Channel Attention Layer
class ChannelAttentionLayer(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttentionLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.global_avg = GlobalAveragePooling2D()
        self.global_max = GlobalMaxPooling2D()
        self.dense1 = Dense(input_shape[-1] // self.ratio, activation="relu")
        self.dense2 = Dense(input_shape[-1], activation="sigmoid")
        super(ChannelAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        avg_out = self.dense2(self.dense1(self.global_avg(inputs)))
        max_out = self.dense2(self.dense1(self.global_max(inputs)))
        attention = Add()([avg_out, max_out])
        return Multiply()([inputs, attention])



# Transformer Encoder Block (Fixed)
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def build(self, input_shape):
        self.seq_len = input_shape[1] if len(input_shape) > 2 else 1  # Ensure valid shape
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=1)  # Add sequence length dimension
        attn_output = self.attention(inputs, inputs)  
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)[:, 0, :]  # Remove extra dim for compatibility



# MAMBA Block (State-Space Model)
class MambaBlock(Layer):
    def __init__(self, units, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation="tanh")
        self.V = Dense(units, activation="sigmoid")
        self.G = Dense(units, activation="relu")

    def call(self, inputs):
        hidden_state = self.W(inputs)
        gate = self.V(inputs)
        new_state = self.G(hidden_state) * gate
        return new_state


# Input
input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)

# Convolutional Layers
x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
x = MaxPool2D(pool_size=(2, 2))(x)

x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.15)(x)

# Apply Channel Attention
x = ChannelAttentionLayer()(x)

# Apply Spatial Attention
x = SpatialAttentionLayer()(x)

x = Flatten()(x)

# Ensure the input dimension is divisible by the number of channels
x = Dense(132, activation="relu")(x)

# Apply Multi-Channel Attention
x = MultiChannelAttentionLayer(num_channels=3)(x)

# Apply Transformer Block
x = TransformerBlock(embed_dim=132, num_heads=4, ff_dim=256)(x)

# Apply MAMBA Block
x = MambaBlock(units=64)(x)

# Fully Connected Layers
x = Dense(units=64, activation="relu")(x)
x = Dropout(rate=0.15)(x)
outputs = Dense(units=7, activation="softmax")(x)
