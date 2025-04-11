
import tensorflow as tf
import keras from tensorflow
import keras_tuner as kt


class AttentionTrainer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = 'models/best_weights_attention.keras'  # Weights-only
        self.callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True, save_weights_only=False),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
        ]

    def build_hypermodel(self, hp):
        """Hypermodel builder for attention network"""
        inputs = keras.layers.Input(shape=self.input_shape)

        # Tunable parameters
        att_units = hp.Int('att_units', 16, 64, step=16)
        branch_units = hp.Int('branch_units', 32, 128, step=32)
        learning_rate = hp.Float('lr', 1e-4, 1e-2, sampling='log')

        # Attention mechanism
        attention = keras.layers.Dense(att_units, activation='softmax')(inputs)
        attended = keras.layers.Multiply()([inputs, attention])

        # Processing branches
        branch1 = keras.layers.Dense(branch_units, activation='swish')(attended)
        branch2 = keras.layers.Dense(branch_units, activation='swish')(attended)

        merged = keras.layers.Concatenate()([branch1, branch2])
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(merged)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                keras.metrics.CategoricalAccuracy(name='precision'),
                keras.metrics.CategoricalAccuracy(name='recall')
            ]
        )
        return model

    def tune_model(self, X_train, y_train, X_val, y_val, max_trials=10):
        """Run hyperparameter tuning"""
        tuner = kt.RandomSearch(
            self.build_hypermodel,
            objective='val_accuracy',
            max_trials=max_trials,
            executions_per_trial=1,
            directory='attention_tuning',
            project_name='iris_attention',
            overwrite=True
        )

        tuner.search(X_train, y_train,
                     epochs=50,
                     validation_data=(X_val, y_val),
                     callbacks=self.callbacks)

        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models()[0]

        return best_model, best_hp
    def compile_model(self, model, learning_rate=0.001):
        # Compile the model with Adam optimizer and sparse categorical loss.
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        return model

    def train(self, model, X_train, y_train, X_val, y_val, epochs=500, batch_size=32):
        self.compile_model(model)
        print("Starting training of the Attention model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=0
        )
        return history

    def hyperparameter_tuning(self, build_model_fn, X_train, y_train, X_val, y_val, max_epochs=50):
        # Use Keras Tuner to find optimal hyperparameters for the attention model.
        tuner = kt.RandomSearch(
            build_model_fn,
            objective='val_sparse_categorical_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory='attention_tuning',
            project_name='AttentionModelTuning'
        )
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=32,
            callbacks=self.callbacks
        )
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters for attention model:", best_hp.values)
        return best_model, best_hp

    def save_model(self, model, path=None):
        save_path = path if path else 'models/final_attention_model.keras'
        model.save(save_path, include_optimizer=False)
        print(f"Attention model successfully saved to {save_path}")
        return save_path

    def load_model(self, path=None):
        load_path = path if path else 'models/final_attention_model.keras'
        try:
            model = keras.models.load_model(load_path, compile=True)
            print(f"Attention model successfully loaded from {load_path}")
            self.compile_model(model)
            return model
        except Exception as e:
            print(f"Error loading attention model from {load_path}: {e}")
            return None