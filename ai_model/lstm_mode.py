"""
Architecture du modèle LSTM pour prédiction des prix Volatility 10
Optimisé pour les séries temporelles financières haute fréquence
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import joblib
import os

from config import config
from data import FeatureSet

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Supprimer les warnings TensorFlow verbeux
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelArchitecture(Enum):
    """Types d'architecture LSTM disponibles"""
    SIMPLE_LSTM = "simple_lstm"
    STACKED_LSTM = "stacked_lstm"
    BIDIRECTIONAL_LSTM = "bidirectional_lstm"
    ATTENTION_LSTM = "attention_lstm"
    HYBRID_LSTM = "hybrid_lstm"


class PredictionType(Enum):
    """Types de prédiction"""
    REGRESSION = "regression"  # Prédiction du prix exact
    CLASSIFICATION = "classification"  # Direction du mouvement
    MULTI_OUTPUT = "multi_output"  # Prix + direction + confiance


@dataclass
class ModelConfig:
    """Configuration du modèle LSTM"""
    # Architecture
    architecture: ModelArchitecture = ModelArchitecture.HYBRID_LSTM
    lstm_units: List[int] = None
    dense_units: List[int] = None
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1

    # Entraînement
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2

    # Régularisation
    l1_reg: float = 0.0
    l2_reg: float = 0.001

    # Données
    lookback_periods: int = 100
    prediction_horizon: int = 5
    prediction_type: PredictionType = PredictionType.MULTI_OUTPUT

    # Features
    use_technical_features: bool = True
    use_price_features: bool = True
    use_volume_features: bool = True
    use_time_features: bool = True
    use_market_features: bool = True

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [64, 32, 16]
        if self.dense_units is None:
            self.dense_units = [32, 16]


@dataclass
class ModelMetrics:
    """Métriques de performance du modèle"""
    # Métriques de régression
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error

    # Métriques de classification
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Métriques financières
    directional_accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Validation
    val_loss: float = 0.0
    val_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'directional_accuracy': self.directional_accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy
        }


class AttentionLayer(layers.Layer):
    """Couche d'attention personnalisée pour LSTM"""

    def __init__(self, attention_dim=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, -1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class LSTMModel:
    """Modèle LSTM pour prédiction des prix Volatility 10"""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = []
        self.model_history = None
        self.metrics = ModelMetrics()

        # Métadonnées
        self.model_info = {
            'version': '1.0.0',
            'created_at': None,
            'last_trained': None,
            'training_samples': 0,
            'input_shape': None,
            'output_shape': None
        }

        # Configuration TensorFlow
        self._configure_tensorflow()

        logger.info(f"Modèle LSTM initialisé avec architecture {self.config.architecture.value}")

    def _configure_tensorflow(self):
        """Configure TensorFlow pour les performances optimales"""
        try:
            # Configuration GPU si disponible
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU détecté et configuré: {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    logger.warning(f"Erreur configuration GPU: {e}")
            else:
                logger.info("Utilisation du CPU pour l'entraînement")

            # Configuration threads
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)

        except Exception as e:
            logger.warning(f"Erreur configuration TensorFlow: {e}")

    def create_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Crée le modèle LSTM selon l'architecture spécifiée

        Args:
            input_shape: Forme des données d'entrée (time_steps, features)

        Returns:
            Modèle Keras compilé
        """
        try:
            logger.info(f"Création du modèle {self.config.architecture.value} avec forme d'entrée {input_shape}")

            if self.config.architecture == ModelArchitecture.SIMPLE_LSTM:
                model = self._create_simple_lstm(input_shape)
            elif self.config.architecture == ModelArchitecture.STACKED_LSTM:
                model = self._create_stacked_lstm(input_shape)
            elif self.config.architecture == ModelArchitecture.BIDIRECTIONAL_LSTM:
                model = self._create_bidirectional_lstm(input_shape)
            elif self.config.architecture == ModelArchitecture.ATTENTION_LSTM:
                model = self._create_attention_lstm(input_shape)
            elif self.config.architecture == ModelArchitecture.HYBRID_LSTM:
                model = self._create_hybrid_lstm(input_shape)
            else:
                raise ValueError(f"Architecture non supportée: {self.config.architecture}")

            # Compiler le modèle
            model = self._compile_model(model)

            # Sauvegarder les métadonnées
            self.model_info['input_shape'] = input_shape
            self.model_info['created_at'] = datetime.now(timezone.utc)

            logger.info(f"Modèle créé avec succès: {model.count_params()} paramètres")
            return model

        except Exception as e:
            logger.error(f"Erreur création du modèle: {e}")
            raise

    def _create_simple_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Crée un modèle LSTM simple"""
        inputs = layers.Input(shape=input_shape, name='input_sequence')

        x = layers.LSTM(
            self.config.lstm_units[0],
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.recurrent_dropout,
            kernel_regularizer=regularizers.l2(self.config.l2_reg),
            name='lstm_layer'
        )(inputs)

        x = layers.Dropout(self.config.dropout_rate)(x)

        # Couches denses
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'dense_{i + 1}'
            )(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

        # Couches de sortie selon le type de prédiction
        outputs = self._create_output_layers(x)

        return Model(inputs=inputs, outputs=outputs, name='SimpleLSTM')

    def _create_stacked_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Crée un modèle LSTM empilé"""
        inputs = layers.Input(shape=input_shape, name='input_sequence')

        x = inputs

        # Couches LSTM empilées
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'lstm_layer_{i + 1}'
            )(x)

            x = layers.Dropout(self.config.dropout_rate)(x)

        # Couches denses
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'dense_{i + 1}'
            )(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = self._create_output_layers(x)

        return Model(inputs=inputs, outputs=outputs, name='StackedLSTM')

    def _create_bidirectional_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Crée un modèle LSTM bidirectionnel"""
        inputs = layers.Input(shape=input_shape, name='input_sequence')

        x = inputs

        # Couches LSTM bidirectionnelles
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout,
                    kernel_regularizer=regularizers.l2(self.config.l2_reg)
                ),
                name=f'bidirectional_lstm_{i + 1}'
            )(x)

            x = layers.Dropout(self.config.dropout_rate)(x)

        # Couches denses
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'dense_{i + 1}'
            )(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = self._create_output_layers(x)

        return Model(inputs=inputs, outputs=outputs, name='BidirectionalLSTM')

    def _create_attention_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Crée un modèle LSTM avec mécanisme d'attention"""
        inputs = layers.Input(shape=input_shape, name='input_sequence')

        # LSTM avec return_sequences=True pour l'attention
        lstm_out = layers.LSTM(
            self.config.lstm_units[0],
            return_sequences=True,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.recurrent_dropout,
            kernel_regularizer=regularizers.l2(self.config.l2_reg),
            name='lstm_layer'
        )(inputs)

        # Couche d'attention
        attention_out = AttentionLayer(
            attention_dim=64,
            name='attention_layer'
        )(lstm_out)

        x = layers.Dropout(self.config.dropout_rate)(attention_out)

        # Couches denses
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'dense_{i + 1}'
            )(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = self._create_output_layers(x)

        return Model(inputs=inputs, outputs=outputs, name='AttentionLSTM')

    def _create_hybrid_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Crée un modèle LSTM hybride (empilé + bidirectionnel + attention)"""
        inputs = layers.Input(shape=input_shape, name='input_sequence')

        # Première couche LSTM bidirectionnelle
        x = layers.Bidirectional(
            layers.LSTM(
                self.config.lstm_units[0],
                return_sequences=True,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                kernel_regularizer=regularizers.l2(self.config.l2_reg)
            ),
            name='bidirectional_lstm_1'
        )(inputs)

        x = layers.Dropout(self.config.dropout_rate)(x)

        # Deuxième couche LSTM
        x = layers.LSTM(
            self.config.lstm_units[1] if len(self.config.lstm_units) > 1 else 32,
            return_sequences=True,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.recurrent_dropout,
            kernel_regularizer=regularizers.l2(self.config.l2_reg),
            name='lstm_layer_2'
        )(x)

        # Couche d'attention
        attention_out = AttentionLayer(
            attention_dim=64,
            name='attention_layer'
        )(x)

        # Connexion résiduelle
        global_pool = layers.GlobalAveragePooling1D()(x)
        combined = layers.concatenate([attention_out, global_pool], name='combined_features')

        x = layers.Dropout(self.config.dropout_rate)(combined)

        # Couches denses avec batch normalization
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'dense_{i + 1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i + 1}')(x)
            x = layers.Activation('relu', name=f'relu_{i + 1}')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = self._create_output_layers(x)

        return Model(inputs=inputs, outputs=outputs, name='HybridLSTM')

    def _create_output_layers(self, x) -> Union[layers.Layer, List[layers.Layer]]:
        """Crée les couches de sortie selon le type de prédiction"""
        if self.config.prediction_type == PredictionType.REGRESSION:
            # Prédiction du prix (régression)
            output = layers.Dense(1, activation='linear', name='price_prediction')(x)
            return output

        elif self.config.prediction_type == PredictionType.CLASSIFICATION:
            # Classification de direction (montée/descente/stable)
            output = layers.Dense(3, activation='softmax', name='direction_prediction')(x)
            return output

        elif self.config.prediction_type == PredictionType.MULTI_OUTPUT:
            # Sorties multiples
            # 1. Prédiction du prix
            price_output = layers.Dense(1, activation='linear', name='price_prediction')(x)

            # 2. Classification de direction
            direction_output = layers.Dense(3, activation='softmax', name='direction_prediction')(x)

            # 3. Confiance de la prédiction
            confidence_output = layers.Dense(1, activation='sigmoid', name='confidence_prediction')(x)

            return [price_output, direction_output, confidence_output]

        else:
            raise ValueError(f"Type de prédiction non supporté: {self.config.prediction_type}")

    def _compile_model(self, model: Model) -> Model:
        """Compile le modèle avec les optimiseurs et métriques appropriées"""
        try:
            # Optimiseur
            optimizer = Adam(
                learning_rate=self.config.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=1.0  # Gradient clipping
            )

            # Loss et métriques selon le type de prédiction
            if self.config.prediction_type == PredictionType.REGRESSION:
                loss = 'mse'
                metrics = ['mae', 'mse']

            elif self.config.prediction_type == PredictionType.CLASSIFICATION:
                loss = 'categorical_crossentropy'
                metrics = ['accuracy', 'precision', 'recall']

            elif self.config.prediction_type == PredictionType.MULTI_OUTPUT:
                loss = {
                    'price_prediction': 'mse',
                    'direction_prediction': 'categorical_crossentropy',
                    'confidence_prediction': 'binary_crossentropy'
                }
                loss_weights = {
                    'price_prediction': 0.4,
                    'direction_prediction': 0.4,
                    'confidence_prediction': 0.2
                }
                metrics = {
                    'price_prediction': ['mae'],
                    'direction_prediction': ['accuracy'],
                    'confidence_prediction': ['binary_accuracy']
                }

                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    loss_weights=loss_weights,
                    metrics=metrics
                )
                return model

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

            return model

        except Exception as e:
            logger.error(f"Erreur compilation du modèle: {e}")
            raise

    def prepare_sequences(self, feature_set: FeatureSet) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les séquences pour l'entraînement LSTM

        Args:
            feature_set: Ensemble de features préparées

        Returns:
            Tuple (X, y) - séquences d'entrée et cibles
        """
        try:
            # Combiner toutes les features
            all_features = pd.concat([
                feature_set.price_features,
                feature_set.technical_features,
                feature_set.volume_features,
                feature_set.time_features,
                feature_set.market_features
            ], axis=1)

            # Filtrer selon la configuration
            selected_features = []

            if self.config.use_price_features and not feature_set.price_features.empty:
                selected_features.append(feature_set.price_features)
            if self.config.use_technical_features and not feature_set.technical_features.empty:
                selected_features.append(feature_set.technical_features)
            if self.config.use_volume_features and not feature_set.volume_features.empty:
                selected_features.append(feature_set.volume_features)
            if self.config.use_time_features and not feature_set.time_features.empty:
                selected_features.append(feature_set.time_features)
            if self.config.use_market_features and not feature_set.market_features.empty:
                selected_features.append(feature_set.market_features)

            if not selected_features:
                raise ValueError("Aucune feature sélectionnée")

            features_df = pd.concat(selected_features, axis=1)
            features_df = features_df.fillna(0)  # Remplacer NaN par 0

            # Préparer les cibles
            targets = feature_set.target_values.fillna(1)  # Classe par défaut = stable

            # Synchroniser les longueurs
            min_length = min(len(features_df), len(targets))
            features_df = features_df.iloc[:min_length]
            targets = targets.iloc[:min_length]

            # Convertir en numpy
            X_data = features_df.values
            y_data = targets.values

            # Créer les séquences
            X, y = self._create_sequences(X_data, y_data)

            # Sauvegarder les noms de features
            self.feature_names = features_df.columns.tolist()

            logger.info(f"Séquences préparées: {X.shape[0]} échantillons, "
                        f"{X.shape[1]} pas de temps, {X.shape[2]} features")

            return X, y

        except Exception as e:
            logger.error(f"Erreur préparation des séquences: {e}")
            raise

    def _create_sequences(self, X_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crée les séquences temporelles pour LSTM"""
        try:
            lookback = self.config.lookback_periods
            horizon = self.config.prediction_horizon

            X_sequences = []
            y_sequences = []

            for i in range(lookback, len(X_data) - horizon + 1):
                # Séquence d'entrée (lookback périodes)
                X_seq = X_data[i - lookback:i]

                # Cible (prédiction à horizon périodes)
                if self.config.prediction_type == PredictionType.REGRESSION:
                    # Prix futur
                    y_seq = X_data[i + horizon - 1, 0]  # Supposer que le prix est la première feature

                elif self.config.prediction_type == PredictionType.CLASSIFICATION:
                    # Direction du mouvement
                    current_price = X_data[i - 1, 0]
                    future_price = X_data[i + horizon - 1, 0]

                    # One-hot encoding: [baisse, stable, hausse]
                    price_change = (future_price - current_price) / current_price
                    if price_change < -0.001:  # Baisse > 0.1%
                        y_seq = [1, 0, 0]
                    elif price_change > 0.001:  # Hausse > 0.1%
                        y_seq = [0, 0, 1]
                    else:  # Stable
                        y_seq = [0, 1, 0]

                elif self.config.prediction_type == PredictionType.MULTI_OUTPUT:
                    # Prix + direction + confiance
                    current_price = X_data[i - 1, 0]
                    future_price = X_data[i + horizon - 1, 0]

                    # Prix
                    price_target = future_price

                    # Direction
                    price_change = (future_price - current_price) / current_price
                    if price_change < -0.001:
                        direction_target = [1, 0, 0]
                    elif price_change > 0.001:
                        direction_target = [0, 0, 1]
                    else:
                        direction_target = [0, 1, 0]

                    # Confiance (basée sur la magnitude du changement)
                    confidence_target = min(1.0, abs(price_change) * 100)

                    y_seq = [price_target, direction_target, confidence_target]

                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

            X_array = np.array(X_sequences)
            y_array = np.array(y_sequences)

            return X_array, y_array

        except Exception as e:
            logger.error(f"Erreur création des séquences: {e}")
            raise

    def get_callbacks(self, model_path: str) -> List:
        """Crée les callbacks pour l'entraînement"""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Réduction du learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Sauvegarde du meilleur modèle
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)

        return callbacks

    def summary(self) -> str:
        """Retourne un résumé du modèle"""
        if self.model is None:
            return "Modèle non créé"

        try:
            summary_lines = []
            self.model.summary(print_fn=summary_lines.append)
            return '\n'.join(summary_lines)
        except Exception as e:
            return f"Erreur génération du résumé: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        info = self.model_info.copy()
        info['config'] = {
            'architecture': self.config.architecture.value,
            'lstm_units': self.config.lstm_units,
            'dense_units': self.config.dense_units,
            'dropout_rate': self.config.dropout_rate,
            'prediction_type': self.config.prediction_type.value,
            'lookback_periods': self.config.lookback_periods,
            'prediction_horizon': self.config.prediction_horizon
        }
        info['metrics'] = self.metrics.to_dict()
        info['feature_count'] = len(self.feature_names)
        info['feature_names'] = self.feature_names

        if self.model:
            info['parameters_count'] = self.model.count_params()
            info['trainable_parameters'] = sum([
                tf.keras.backend.count_params(w) for w in self.model.trainable_weights
            ])

        return info


# Instance globale avec configuration par défaut
default_config = ModelConfig(
    architecture=ModelArchitecture(
        config.ai_model.use_price_action and ModelArchitecture.HYBRID_LSTM.value or ModelArchitecture.STACKED_LSTM.value),
    lstm_units=config.ai_model.lstm_units,
    dropout_rate=config.ai_model.dropout_rate,
    learning_rate=config.ai_model.learning_rate,
    batch_size=config.ai_model.batch_size,
    epochs=config.ai_model.epochs,
    validation_split=config.ai_model.validation_split,
    lookback_periods=config.ai_model.lookback_periods,
    prediction_horizon=config.ai_model.prediction_horizon,
    use_technical_indicators=config.ai_model.use_technical_indicators,
    use_price_action=config.ai_model.use_price_action,
    use_volume=config.ai_model.use_volume,
    normalize_features=config.ai_model.normalize_features
)

lstm_model = LSTMModel(default_config)

if __name__ == "__main__":
    # Test du modèle LSTM
    print("🧠 Test du modèle LSTM...")

    try:
        # Configuration de test
        test_config = ModelConfig(
            architecture=ModelArchitecture.HYBRID_LSTM,
            lstm_units=[32, 16],
            dense_units=[16, 8],
            lookback_periods=50,
            prediction_horizon=3,
            epochs=5  # Peu d'époques pour le test
        )

        model = LSTMModel(test_config)

        # Créer un modèle de test
        input_shape = (50, 20)  # 50 pas de temps, 20 features
        keras_model = model.create_model(input_shape)

        print(f"✅ Modèle créé:")
        print(f"   Architecture: {test_config.architecture.value}")
        print(f"   Paramètres: {keras_model.count_params()}")
        print(f"   Forme d'entrée: {input_shape}")

        # Informations du modèle
        model_info = model.get_model_info()
        print(f"\n📊 Informations du modèle:")
        print(f"   Type de prédiction: {model_info['config']['prediction_type']}")
        print(f"   Lookback: {model_info['config']['lookback_periods']}")
        print(f"   Horizon: {model_info['config']['prediction_horizon']}")

        # Test avec données synthétiques
        print(f"\n🔬 Test avec données synthétiques...")
        X_test = np.random.randn(100, 50, 20)

        if test_config.prediction_type == PredictionType.MULTI_OUTPUT:
            y_pred = keras_model.predict(X_test, verbose=0)
            print(f"   Prédictions multiples: {len(y_pred)} sorties")
            print(f"   Forme prix: {y_pred[0].shape}")
            print(f"   Forme direction: {y_pred[1].shape}")
            print(f"   Forme confiance: {y_pred[2].shape}")
        else:
            y_pred = keras_model.predict(X_test, verbose=0)
            print(f"   Forme de sortie: {y_pred.shape}")

        print("✅ Test du modèle LSTM réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test du modèle LSTM échoué: {e}")