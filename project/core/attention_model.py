import tensorflow
import keras from tensorflow

def create_attention_model(input_shape, num_classes, att_units=None, branch_units=64):
    """Define an attention-based neural network for tabular data (Iris features)."""
    inputs = keras.layers.Input(shape=input_shape)
    # Attention mechanism: learn a weight for each feature
    att_units = att_units if att_units is not None else input_shape[-1]  # default to number of features
    attention_weights = keras.layers.Dense(att_units, activation='softmax')(inputs)
    attended_features = keras.layers.Multiply()([inputs, attention_weights])  # weight features by attention

    # Two parallel branches process the attended features
    branch1 = keras.layers.Dense(branch_units, activation='swish')(attended_features)
    branch2 = keras.layers.Dense(branch_units, activation='swish')(attended_features)

    # Concatenate branch outputs and produce final class probabilities
    merged = keras.layers.Concatenate()([branch1, branch2])
    outputs = keras.layers.Dense(num_classes, activation='softmax')(merged)

    return keras.models.Model(inputs=inputs, outputs=outputs)