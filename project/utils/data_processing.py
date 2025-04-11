import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class IrisDataEngineer:
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def _load_structured_data(self):
        """Data loading with biological feature analysis"""
        iris = load_iris()
        df = pd.DataFrame(iris.data,
                        columns=[name.replace(' (cm)', '') for name in iris.feature_names])
        df['species'] = iris.target_names[iris.target]
        return df

    def _stratified_split(self, X, y):
        """Maintain species distribution across splits"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size,
            stratify=y, random_state=self.random_state
        )

        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            stratify=y_temp, random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def process(self):
            """Preprocessing pipeline"""
            raw_df = self._load_structured_data()
            X = raw_df.drop('species', axis=1)
            y = raw_df['species'].astype('category').cat.codes

            # Standard stratified splitting
            X_train, X_val, X_test, y_train, y_val, y_test = self._stratified_split(X, y)

            # Feature engineering
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

            # Ensure labels remain integers
            y_train = y_train.astype(np.int32)
            y_val = y_val.astype(np.int32)
            y_test = y_test.astype(np.int32)

            # Data preservation
            import os
            if not os.path.exists('data'):
                os.makedirs('data')

            np.save('data/X_train.npy', X_train)
            np.save('data/X_val.npy', X_val)
            np.save('data/X_test.npy', X_test)
            np.save('data/y_train.npy', y_train)
            np.save('data/y_val.npy', y_val)
            np.save('data/y_test.npy', y_test)

            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
