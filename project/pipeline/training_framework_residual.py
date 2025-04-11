import tensorflow
import keras from tensorflow
import keras_tuner as kt


class ResidualTrainer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = 'models/best_weights_residual.keras'
        self.callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True, save_weights_only=False),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
        ]

    def build_hypermodel(self, hp):
        """Hypermodel builder for residual network"""
        inputs = keras.layers.Input(shape=self.input_shape)

        # Tunable parameters
        base_units = hp.Int('units', 64, 256, step=64)
        learning_rate = hp.Float('lr', 1e-4, 1e-2, sampling='log')

        # Residual blocks
        x = keras.layers.Dense(base_units, activation='swish', kernel_initializer='he_normal')(inputs)
        for _ in range(3):
            shortcut = x
            x = keras.layers.Dense(base_units, activation='swish')(x)
            x = keras.layers.Add()([shortcut, x])
            x = keras.layers.Activation('swish')(x)

        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name='accuracy')
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
            directory='residual_tuning',
            project_name='iris_residual',
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
        """Compile the model with Adam optimizer"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        return model

    def train(self, model, X_train, y_train, X_val, y_val, epochs=500, batch_size=32):
        self.compile_model(model)
        print("Starting training of the Residual model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=0
        )
        return history

    def save_model(self, model, path=None):
        save_path = path if path else 'models/final_residual_model.keras'
        model.save(save_path, include_optimizer=False)
        print(f"Residual model successfully saved to {save_path}")
        return save_path

    def load_model(self, path=None):
        load_path = path if path else 'models/final_residual_model.keras'
        try:
            model = keras.models.load_model(load_path, compile=True)
            print(f"Residual model successfully loaded from {load_path}")
            self.compile_model(model)
            return model
        except Exception as e:
            print(f"Error loading residual model from {load_path}: {e}")
            return None
