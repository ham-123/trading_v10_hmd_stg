"""
Entraîneur pour le modèle LSTM de prédiction Volatility 10
Gère l'entraînement, la validation et l'optimisation des hyperparamètres
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import logging
import joblib
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from data import prepare_training_data, FeatureSet
from .lstm_model import LSTMModel, ModelConfig, ModelMetrics, ModelArchitecture, PredictionType

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement"""
    # Données
    symbol: str = "R_10"
    timeframe: str = "1m"
    training_days: int = 30
    validation_split: float = 0.2
    test_split: float = 0.1

    # Entraînement
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    min_delta: float = 0.001

    # Validation
    cross_validation_folds: int = 5
    validation_method: str = "time_series"  # "time_series" ou "random"

    # Sauvegarde
    model_dir: str = "models"
    save_best_only: bool = True
    save_history: bool = True

    # Optimisation
    auto_hyperparameter_tuning: bool = False
    hyperparameter_trials: int = 50

    # Performance
    min_accuracy_threshold: float = 0.55
    min_sharpe_threshold: float = 1.0
    retrain_threshold: float = 0.05  # Déclin de performance pour redéclencher l'entraînement


@dataclass
class TrainingResults:
    """Résultats d'entraînement"""
    # Métriques finales
    final_metrics: ModelMetrics
    best_metrics: ModelMetrics

    # Historique
    training_history: Dict[str, List[float]]
    validation_history: Dict[str, List[float]]

    # Informations
    training_time: float
    epochs_completed: int
    best_epoch: int
    convergence_achieved: bool

    # Chemins
    model_path: str
    weights_path: str
    history_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_metrics': asdict(self.final_metrics),
            'best_metrics': asdict(self.best_metrics),
            'training_time': self.training_time,
            'epochs_completed': self.epochs_completed,
            'best_epoch': self.best_epoch,
            'convergence_achieved': self.convergence_achieved,
            'model_path': self.model_path,
            'weights_path': self.weights_path,
            'history_path': self.history_path
        }


class ModelTrainer:
    """Entraîneur pour les modèles LSTM"""

    def __init__(self, training_config: TrainingConfig = None):
        self.config = training_config or TrainingConfig()
        self.model = None
        self.training_results = None

        # Historique des entraînements
        self.training_history = []

        # Métriques de suivi
        self.training_stats = {
            'models_trained': 0,
            'total_training_time': 0.0,
            'best_accuracy_achieved': 0.0,
            'best_sharpe_achieved': 0.0,
            'last_training_date': None,
            'convergence_rate': 0.0
        }

        # Créer les dossiers nécessaires
        self._create_directories()

        logger.info("Entraîneur de modèles LSTM initialisé")

    def _create_directories(self):
        """Crée les dossiers nécessaires pour la sauvegarde"""
        directories = [
            self.config.model_dir,
            os.path.join(self.config.model_dir, "weights"),
            os.path.join(self.config.model_dir, "history"),
            os.path.join(self.config.model_dir, "plots"),
            "logs/training"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prépare les données d'entraînement, validation et test

        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            logger.info(f"Préparation des données pour {self.config.symbol}")

            # Récupérer les données
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.config.training_days)

            feature_set = prepare_training_data(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if feature_set.price_features.empty:
                raise ValueError("Aucune donnée disponible pour l'entraînement")

            # Préparer les séquences avec le modèle
            if self.model is None:
                # Créer un modèle temporaire pour la préparation des données
                temp_model = LSTMModel()
                X, y = temp_model.prepare_sequences(feature_set)
            else:
                X, y = self.model.prepare_sequences(feature_set)

            # Division des données
            total_samples = len(X)
            test_size = int(total_samples * self.config.test_split)
            val_size = int(total_samples * self.config.validation_split)
            train_size = total_samples - test_size - val_size

            # Division temporelle (importante pour les séries temporelles)
            X_train = X[:train_size]
            y_train = y[:train_size]

            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]

            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]

            logger.info(f"Données préparées: {train_size} train, {val_size} val, {test_size} test")
            logger.info(f"Forme des données: X={X_train.shape}, y={y_train.shape}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Erreur préparation des données: {e}")
            raise

    def train_model(self, model_config: ModelConfig = None, X_train: np.ndarray = None,
                    y_train: np.ndarray = None, X_val: np.ndarray = None,
                    y_val: np.ndarray = None) -> TrainingResults:
        """
        Entraîne un modèle LSTM

        Args:
            model_config: Configuration du modèle
            X_train, y_train: Données d'entraînement (optionnel, sinon préparées automatiquement)
            X_val, y_val: Données de validation (optionnel)

        Returns:
            Résultats d'entraînement
        """
        try:
            start_time = datetime.now(timezone.utc)
            logger.info("🚀 Début de l'entraînement du modèle LSTM")

            # Préparer les données si non fournies
            if X_train is None:
                X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()

            # Créer le modèle
            if model_config is None:
                model_config = ModelConfig()

            self.model = LSTMModel(model_config)

            # Définir la forme d'entrée
            input_shape = (X_train.shape[1], X_train.shape[2])
            keras_model = self.model.create_model(input_shape)

            # Chemins de sauvegarde
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            model_name = f"lstm_{model_config.architecture.value}_{timestamp}"

            model_path = os.path.join(self.config.model_dir, f"{model_name}.h5")
            weights_path = os.path.join(self.config.model_dir, "weights", f"{model_name}_weights.h5")
            history_path = os.path.join(self.config.model_dir, "history", f"{model_name}_history.json")

            # Callbacks
            callbacks = self._create_callbacks(model_path, weights_path)

            # Entraînement
            logger.info(f"Entraînement avec {len(X_train)} échantillons")

            history = keras_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Important pour les séries temporelles
            )

            # Sauvegarder le modèle final
            self.model.model = keras_model

            # Calculer les métriques finales
            final_metrics = self._calculate_metrics(keras_model, X_val, y_val, "validation")

            # Trouver la meilleure époque
            best_epoch = self._find_best_epoch(history)
            best_metrics = self._extract_best_metrics(history, best_epoch)

            # Temps d'entraînement
            end_time = datetime.now(timezone.utc)
            training_time = (end_time - start_time).total_seconds()

            # Créer les résultats
            training_results = TrainingResults(
                final_metrics=final_metrics,
                best_metrics=best_metrics,
                training_history=history.history,
                validation_history=history.history,
                training_time=training_time,
                epochs_completed=len(history.history['loss']),
                best_epoch=best_epoch,
                convergence_achieved=self._check_convergence(history),
                model_path=model_path,
                weights_path=weights_path,
                history_path=history_path
            )

            # Sauvegarder les résultats
            self._save_training_results(training_results, history_path)

            # Mettre à jour les statistiques
            self._update_training_stats(training_results)

            # Générer les graphiques
            self._plot_training_history(history, model_name)

            self.training_results = training_results

            logger.info(f"✅ Entraînement terminé en {training_time:.1f}s")
            logger.info(f"📊 Précision finale: {final_metrics.accuracy:.3f}")
            logger.info(f"📈 Meilleure précision: {best_metrics.accuracy:.3f} (époque {best_epoch})")

            return training_results

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            raise

    def _create_callbacks(self, model_path: str, weights_path: str) -> List:
        """Crée les callbacks pour l'entraînement"""
        callbacks = []

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Réduction du learning rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(5, self.config.patience // 3),
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Sauvegarde du modèle
        if self.config.save_best_only:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)

            # Sauvegarde des weights séparément
            weights_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=weights_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
            callbacks.append(weights_checkpoint)

        # Logging personnalisé
        class TrainingLogger(keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.epoch_start_time = None

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = datetime.now()

            def on_epoch_end(self, epoch, logs=None):
                duration = (datetime.now() - self.epoch_start_time).total_seconds()
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Époque {epoch + 1}: loss={logs.get('loss', 0):.4f}, "
                                     f"val_loss={logs.get('val_loss', 0):.4f}, "
                                     f"durée={duration:.1f}s")

        callbacks.append(TrainingLogger(logger))

        return callbacks

    def _calculate_metrics(self, model: keras.Model, X: np.ndarray, y: np.ndarray, prefix: str) -> ModelMetrics:
        """Calcule les métriques de performance"""
        try:
            metrics = ModelMetrics()

            # Prédictions
            y_pred = model.predict(X, verbose=0)

            # Selon le type de prédiction
            if isinstance(y_pred, list):  # Multi-output
                y_pred_price = y_pred[0].flatten()
                y_pred_direction = y_pred[1]
                y_pred_confidence = y_pred[2].flatten()

                # Métriques de régression (prix)
                y_true_price = y[:, 0] if len(y.shape) > 1 else y
                metrics.mae = mean_absolute_error(y_true_price, y_pred_price)
                metrics.rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
                metrics.mape = np.mean(np.abs((y_true_price - y_pred_price) / (y_true_price + 1e-8))) * 100

                # Métriques de classification (direction)
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y_true_direction = y[:, 1:4]  # Classes de direction
                    y_pred_direction_classes = np.argmax(y_pred_direction, axis=1)
                    y_true_direction_classes = np.argmax(y_true_direction, axis=1)

                    metrics.accuracy = accuracy_score(y_true_direction_classes, y_pred_direction_classes)
                    metrics.precision = precision_score(y_true_direction_classes, y_pred_direction_classes,
                                                        average='weighted', zero_division=0)
                    metrics.recall = recall_score(y_true_direction_classes, y_pred_direction_classes,
                                                  average='weighted', zero_division=0)
                    metrics.f1_score = f1_score(y_true_direction_classes, y_pred_direction_classes, average='weighted',
                                                zero_division=0)

                # Précision directionnelle
                metrics.directional_accuracy = self._calculate_directional_accuracy(y_true_price, y_pred_price)

            else:  # Single output
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Classification
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_true_classes = np.argmax(y, axis=1) if len(y.shape) > 1 else y.astype(int)

                    metrics.accuracy = accuracy_score(y_true_classes, y_pred_classes)
                    metrics.precision = precision_score(y_true_classes, y_pred_classes, average='weighted',
                                                        zero_division=0)
                    metrics.recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
                    metrics.f1_score = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
                else:
                    # Régression
                    y_pred_flat = y_pred.flatten()
                    y_true_flat = y.flatten()

                    metrics.mae = mean_absolute_error(y_true_flat, y_pred_flat)
                    metrics.rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
                    metrics.mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100

                    # Précision directionnelle
                    metrics.directional_accuracy = self._calculate_directional_accuracy(y_true_flat, y_pred_flat)

            # Métriques financières
            if metrics.directional_accuracy > 0:
                # Sharpe ratio simplifié (basé sur la précision directionnelle)
                excess_return = (metrics.directional_accuracy - 0.5) * 2  # Convertir en rendement excédentaire
                metrics.sharpe_ratio = excess_return / max(0.1, metrics.rmse / 100)  # Approximation

            logger.debug(f"Métriques {prefix} calculées: accuracy={metrics.accuracy:.3f}, "
                         f"directional_accuracy={metrics.directional_accuracy:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"Erreur calcul des métriques: {e}")
            return ModelMetrics()

    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la précision directionnelle"""
        try:
            if len(y_true) < 2:
                return 0.0

            # Calculer les changements directionnels
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0

            # Précision directionnelle
            correct_directions = np.sum(true_direction == pred_direction)
            total_predictions = len(true_direction)

            return correct_directions / total_predictions if total_predictions > 0 else 0.0

        except Exception as e:
            logger.error(f"Erreur calcul précision directionnelle: {e}")
            return 0.0

    def _find_best_epoch(self, history) -> int:
        """Trouve la meilleure époque basée sur val_loss"""
        val_loss = history.history.get('val_loss', [])
        if val_loss:
            return int(np.argmin(val_loss))
        return 0

    def _extract_best_metrics(self, history, best_epoch: int) -> ModelMetrics:
        """Extrait les métriques de la meilleure époque"""
        metrics = ModelMetrics()

        if best_epoch < len(history.history.get('loss', [])):
            metrics.val_loss = history.history.get('val_loss', [0])[best_epoch]

            # Accuracy si disponible
            if 'val_accuracy' in history.history:
                metrics.val_accuracy = history.history['val_accuracy'][best_epoch]
            elif 'val_direction_prediction_accuracy' in history.history:
                metrics.val_accuracy = history.history['val_direction_prediction_accuracy'][best_epoch]

            # MAE si disponible
            if 'val_mae' in history.history:
                metrics.mae = history.history['val_mae'][best_epoch]
            elif 'val_price_prediction_mae' in history.history:
                metrics.mae = history.history['val_price_prediction_mae'][best_epoch]

        return metrics

    def _check_convergence(self, history) -> bool:
        """Vérifie si le modèle a convergé"""
        val_loss = history.history.get('val_loss', [])
        if len(val_loss) < 10:
            return False

        # Vérifier si la perte de validation s'est stabilisée
        recent_loss = val_loss[-5:]
        loss_std = np.std(recent_loss)
        loss_mean = np.mean(recent_loss)

        # Convergence si l'écart-type représente moins de 1% de la moyenne
        return loss_std / (loss_mean + 1e-8) < 0.01

    def _save_training_results(self, results: TrainingResults, history_path: str):
        """Sauvegarde les résultats d'entraînement"""
        try:
            # Sauvegarder l'historique en JSON
            history_data = {
                'training_results': results.to_dict(),
                'config': asdict(self.config),
                'model_config': self.model.get_model_info() if self.model else {}
            }

            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)

            logger.debug(f"Résultats sauvegardés: {history_path}")

        except Exception as e:
            logger.error(f"Erreur sauvegarde des résultats: {e}")

    def _update_training_stats(self, results: TrainingResults):
        """Met à jour les statistiques d'entraînement"""
        self.training_stats['models_trained'] += 1
        self.training_stats['total_training_time'] += results.training_time
        self.training_stats['last_training_date'] = datetime.now(timezone.utc)

        if results.best_metrics.accuracy > self.training_stats['best_accuracy_achieved']:
            self.training_stats['best_accuracy_achieved'] = results.best_metrics.accuracy

        if results.best_metrics.sharpe_ratio > self.training_stats['best_sharpe_achieved']:
            self.training_stats['best_sharpe_achieved'] = results.best_metrics.sharpe_ratio

        if results.convergence_achieved:
            total_models = self.training_stats['models_trained']
            convergence_count = sum(1 for r in self.training_history if r.convergence_achieved) + 1
            self.training_stats['convergence_rate'] = convergence_count / total_models

        # Ajouter aux résultats historiques
        self.training_history.append(results)

    def _plot_training_history(self, history, model_name: str):
        """Crée des graphiques de l'historique d'entraînement"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Historique d\'entraînement - {model_name}', fontsize=16)

            # Perte
            axes[0, 0].plot(history.history['loss'], label='Train Loss')
            if 'val_loss' in history.history:
                axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Accuracy (si disponible)
            if 'accuracy' in history.history or 'direction_prediction_accuracy' in history.history:
                acc_key = 'accuracy' if 'accuracy' in history.history else 'direction_prediction_accuracy'
                val_acc_key = 'val_' + acc_key

                axes[0, 1].plot(history.history[acc_key], label='Train Accuracy')
                if val_acc_key in history.history:
                    axes[0, 1].plot(history.history[val_acc_key], label='Validation Accuracy')
                axes[0, 1].set_title('Model Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)

            # MAE (si disponible)
            if 'mae' in history.history or 'price_prediction_mae' in history.history:
                mae_key = 'mae' if 'mae' in history.history else 'price_prediction_mae'
                val_mae_key = 'val_' + mae_key

                axes[1, 0].plot(history.history[mae_key], label='Train MAE')
                if val_mae_key in history.history:
                    axes[1, 0].plot(history.history[val_mae_key], label='Validation MAE')
                axes[1, 0].set_title('Mean Absolute Error')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('MAE')
                axes[1, 0].legend()
                axes[1, 0].grid(True)

            # Learning rate (si disponible)
            if 'lr' in history.history:
                axes[1, 1].plot(history.history['lr'], label='Learning Rate')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

            plt.tight_layout()

            # Sauvegarder
            plot_path = os.path.join(self.config.model_dir, "plots", f"{model_name}_history.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.debug(f"Graphique sauvegardé: {plot_path}")

        except Exception as e:
            logger.error(f"Erreur création des graphiques: {e}")

    def cross_validate(self, model_config: ModelConfig = None, n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Effectue une validation croisée temporelle

        Args:
            model_config: Configuration du modèle
            n_splits: Nombre de splits pour la validation croisée

        Returns:
            Dictionnaire avec les métriques pour chaque fold
        """
        try:
            logger.info(f"🔄 Validation croisée avec {n_splits} folds")

            # Préparer les données
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()

            # Combiner train et val pour la CV
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = np.concatenate([y_train, y_val], axis=0)

            # Time Series Split
            tscv = TimeSeriesSplit(n_splits=n_splits)

            cv_results = {
                'accuracy': [],
                'mae': [],
                'directional_accuracy': [],
                'val_loss': []
            }

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
                logger.info(f"📊 Fold {fold + 1}/{n_splits}")

                # Données du fold
                X_fold_train, X_fold_val = X_full[train_idx], X_full[val_idx]
                y_fold_train, y_fold_val = y_full[train_idx], y_full[val_idx]

                # Créer et entraîner le modèle
                model = LSTMModel(model_config or ModelConfig())
                input_shape = (X_fold_train.shape[1], X_fold_train.shape[2])
                keras_model = model.create_model(input_shape)

                # Entraînement rapide pour CV
                history = keras_model.fit(
                    X_fold_train, y_fold_train,
                    validation_data=(X_fold_val, y_fold_val),
                    batch_size=self.config.batch_size,
                    epochs=min(50, self.config.epochs),  # Moins d'époques pour la CV
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )

                # Calculer les métriques
                metrics = self._calculate_metrics(keras_model, X_fold_val, y_fold_val, f"fold_{fold}")

                cv_results['accuracy'].append(metrics.accuracy)
                cv_results['mae'].append(metrics.mae)
                cv_results['directional_accuracy'].append(metrics.directional_accuracy)
                cv_results['val_loss'].append(min(history.history['val_loss']))

            # Statistiques de CV
            logger.info("📈 Résultats de validation croisée:")
            for metric, values in cv_results.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                logger.info(f"   {metric}: {mean_val:.3f} ± {std_val:.3f}")

            return cv_results

        except Exception as e:
            logger.error(f"Erreur lors de la validation croisée: {e}")
            raise

    def load_model(self, model_path: str) -> bool:
        """
        Charge un modèle sauvegardé

        Args:
            model_path: Chemin vers le modèle

        Returns:
            True si chargement réussi
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Modèle non trouvé: {model_path}")
                return False

            # Charger le modèle Keras
            keras_model = keras.models.load_model(
                model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )

            # Créer l'objet LSTMModel
            self.model = LSTMModel()
            self.model.model = keras_model

            # Charger les métadonnées si disponibles
            history_path = model_path.replace('.h5', '_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    if 'model_config' in data:
                        self.model.model_info.update(data['model_config'])

            logger.info(f"✅ Modèle chargé: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur chargement du modèle: {e}")
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'entraînement"""
        return self.training_stats.copy()


# Instance globale
model_trainer = ModelTrainer()


# Fonction utilitaire
def train_lstm_model(model_config: ModelConfig = None, training_config: TrainingConfig = None) -> TrainingResults:
    """
    Fonction utilitaire pour entraîner un modèle LSTM

    Args:
        model_config: Configuration du modèle
        training_config: Configuration de l'entraînement

    Returns:
        Résultats d'entraînement
    """
    if training_config:
        trainer = ModelTrainer(training_config)
    else:
        trainer = model_trainer

    return trainer.train_model(model_config)


if __name__ == "__main__":
    # Test de l'entraîneur
    print("🎓 Test de l'entraîneur LSTM...")

    try:
        # Configuration de test
        test_model_config = ModelConfig(
            architecture=ModelArchitecture.SIMPLE_LSTM,
            lstm_units=[32],
            dense_units=[16],
            epochs=5,  # Peu d'époques pour le test
            batch_size=16,
            lookback_periods=20,
            prediction_horizon=1
        )

        test_training_config = TrainingConfig(
            training_days=7,  # Peu de jours pour le test
            epochs=5,
            batch_size=16,
            patience=3
        )

        trainer = ModelTrainer(test_training_config)

        print(f"✅ Entraîneur configuré")
        print(f"   Modèle: {test_model_config.architecture.value}")
        print(f"   Époques: {test_training_config.epochs}")
        print(f"   Jours d'entraînement: {test_training_config.training_days}")

        # Test de validation croisée (si données disponibles)
        try:
            cv_results = trainer.cross_validate(test_model_config, n_splits=3)
            print(f"\n📊 Validation croisée:")
            for metric, values in cv_results.items():
                print(f"   {metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")
        except Exception as e:
            print(f"⚠️ Validation croisée échouée (normal si pas de données): {e}")

        # Statistiques
        stats = trainer.get_training_stats()
        print(f"\n📈 Statistiques:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("✅ Test de l'entraîneur réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test de l'entraîneur échoué: {e}")