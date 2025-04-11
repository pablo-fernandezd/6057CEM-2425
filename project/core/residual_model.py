import tensorflow
import keras from tensorflow

def create_deep_residual_model(input_shape, num_classes, base_units=128, num_blocks=3):
    """Define a deep residual network for tabular data (Iris features)."""
    inputs = keras.layers.Input(shape=input_shape)
    # Initial dense layer to project input to a higher-dimensional feature space
    x = keras.layers.Dense(base_units, activation='swish', kernel_initializer='he_normal')(inputs)
    # Residual blocks: each adds a skip connection around a dense layer
    for _ in range(num_blocks):
        shortcut = x  # save input to block
        x = keras.layers.Dense(base_units, activation='swish')(x)
        x = keras.layers.Add()([shortcut, x])            # add skip-connection
        x = keras.layers.Activation('swish')(x)          # apply activation after addition
    # Output layer for class probabilities
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.models.Model(inputs=inputs, outputs=outputs)