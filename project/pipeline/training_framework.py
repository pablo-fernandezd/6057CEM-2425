import keras_tuner as kt
import tensorflow
import keras from tensorflow



class ModelTrainer:
    def __init__(self, architecture, input_shape, num_classes):
        """
        Initialize the trainer for a given architecture.
        architecture: str, "residual" or "attention"
        """
        self.architecture = architecture.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = f'models/best_model_{self.architecture}.keras'
        self.callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True, save_weights_only=False),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
        ]

    def create_model(self):
        """Build the model architecture (residual or attention) with default parameters."""
        if self.architecture == 'residual':
            # Import and build residual model
            # Ignore IDE syntax warning, being called from the notebook in the parent folder will be allright
            from core.residual_model import create_deep_residual_model
            model = create_deep_residual_model(self.input_shape, self.num_classes)
        elif self.architecture == 'attention':
            # Import and build attention model
            # Ignore IDE syntax warning, being called from the notebook in the parent folder will be allright
            from core.attention_model import create_attention_model
            model = create_attention_model(self.input_shape, self.num_classes)
        else:
            raise ValueError(f"Unknown architecture type: {self.architecture}")
        return model

    def create_model_with_hyperparameters(self, hp_values):
        """Create a new model instance using the best hyperparameters."""
        if self.architecture == 'residual':
            # Ignore IDE syntax warning, being called from the notebook in the parent folder will be allright
            from core.residual_model import create_deep_residual_model
            # Extract hyperparameters for residual model
            units = hp_values['units']
            model = create_deep_residual_model(self.input_shape, self.num_classes, base_units=units)
        else:  # attention model
            # Ignore IDE syntax warning, being called from the notebook in the parent folder will be allright
            from core.attention_model import create_attention_model
            # Extract hyperparameters for attention model
            branch_units = hp_values['branch_units']
            model = create_attention_model(self.input_shape, self.num_classes, branch_units=branch_units)

        return model

    def compile_model(self, model, learning_rate=0.001):
        """Compile the model with Adam optimizer and appropriate metrics."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']  # using sparse categorical accuracy for multi-class
        )
        return model

    def train(self, model, X_train, y_train, X_val, y_val, epochs=500, batch_size=32):
        """Compile and train the model, returning the training history."""
        # Compile the model
        self.compile_model(model)
        # Logging the start of training
        print(f"\n🔥 Training the {self.architecture.capitalize()} model...")
        print("Training details:")
        print(f"- Epochs: {epochs}, Batch size: {batch_size}")
        print(f"- Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        # Fit the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=0  # Suppress per-epoch output for brevity (early stopping will restore best weights)
        )
        print(
            f"✅ {self.architecture.capitalize()} model training complete. Best epoch: {len(history.history['loss'])} with validation accuracy {history.history.get('val_sparse_categorical_accuracy', [None])[-1]:.4f}")
        return history

    def tune_model(self, X_train, y_train, X_val, y_val, max_trials=10, epochs=50):
        """Use Keras Tuner to find optimal hyperparameters for the model."""
        print(f"\n🔧 Hyperparameter Tuning for {self.architecture.capitalize()} model...")

        # Define the model builder for Keras Tuner
        def model_builder(hp):
            inputs = keras.layers.Input(shape=self.input_shape)
            if self.architecture == 'residual':
                # Hyperparameters for residual model
                units = hp.Int('units', min_value=64, max_value=256, step=64)
                lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
                # Build residual architecture with tunable units
                x = keras.layers.Dense(units, activation='swish', kernel_initializer='he_normal')(inputs)
                # Use a fixed number of residual blocks (e.g., 3), but with tunable layer size
                for _ in range(3):
                    shortcut = x
                    x = keras.layers.Dense(units, activation='swish')(x)
                    x = keras.layers.Add()([shortcut, x])
                    x = keras.layers.Activation('swish')(x)
                outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
                model = keras.Model(inputs, outputs)
            else:  # attention model
                # Hyperparameters for attention model
                branch_units = hp.Int('branch_units', min_value=32, max_value=128, step=32)
                lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
                # Build attention architecture (attention vector size fixed to input dimension for shape alignment)
                attention = keras.layers.Dense(self.input_shape[-1], activation='softmax')(inputs)
                attended = keras.layers.Multiply()([inputs, attention])
                branch1 = keras.layers.Dense(branch_units, activation='swish')(attended)
                branch2 = keras.layers.Dense(branch_units, activation='swish')(attended)
                merged = keras.layers.Concatenate()([branch1, branch2])
                outputs = keras.layers.Dense(self.num_classes, activation='softmax')(merged)
                model = keras.Model(inputs, outputs)
            # Compile the model with the hyperparameter-specific learning rate
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy']
            )
            return model

        # Set up the Keras Tuner
        tuner = kt.RandomSearch(
            model_builder,
            objective='val_sparse_categorical_accuracy',
            max_trials=max_trials,
            executions_per_trial=1,
            directory=f'{self.architecture}_tuning',
            project_name=f'iris_{self.architecture}',
            overwrite = True
        )

        # Perform the search
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=self.callbacks,
            verbose=0
        )

        # Retrieve the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("   Best hyperparameters found:", best_hp.values)

        # Create a fresh model with the best hyperparameters instead of using the tuner model
        fresh_model = self.create_model_with_hyperparameters(best_hp.values)

        # Compile with the best learning rate
        learning_rate = best_hp.values['learning_rate']
        self.compile_model(fresh_model, learning_rate=learning_rate)

        return fresh_model, best_hp

    def save_model(self, model, path=None):
        """Save the model to disk."""
        save_path = path if path else f'models/final_{self.architecture}_model.keras'
        model.save(save_path, include_optimizer=False)
        print(f"💾 {self.architecture.capitalize()} model saved to {save_path}")
        return save_path

    def load_model(self, path=None):
        """Load the model from disk (with compiled state)."""
        load_path = path if path else f'models/final_{self.architecture}_model.keras'
        try:
            # Load without compiling first
            model = keras.models.load_model(load_path, compile=False)
            print(f"🔄 {self.architecture.capitalize()} model loaded from {load_path}")
            # Then compile manually
            self.compile_model(model)
            return model
        except Exception as e:
            print(f"Error loading {self.architecture} model from {load_path}: {e}")
            return None
