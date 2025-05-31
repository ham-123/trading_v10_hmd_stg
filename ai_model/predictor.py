"""
Prédicteur temps réel utilisant les modèles LSTM entraînés
Optimisé pour les prédictions haute fréquence sur Volatility 10
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
from threading import Lock
import time

from config import config
from data import db_manager, prepare_training_data, FeatureSet
from .lstm_model import LSTMModel, ModelConfig, ModelMetrics, PredictionType
from .trainer import ModelTrainer

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Résultat d'une prédiction"""
    # Prédiction principale
    predicted_price: Optional[float] = None
    predicted_direction: Optional[str] = None  # 'UP', 'DOWN', 'STABLE'
    confidence: float = 0.0

    # Prédictions détaillées
    direction_probabilities: Optional[Dict[str, float]] = None  # {'DOWN': 0.2, 'STABLE': 0.3, 'UP': 0.5}
    price_change_pct: Optional[float] = None
    volatility_prediction: Optional[float] = None

    # Contexte
    current_price: float = 0.0
    prediction_horizon: int = 5  # minutes
    timestamp: datetime = None

    # Qualité de la prédiction
    model_confidence: float = 0.0
    feature_quality_score: float = 0.0
    prediction_reliability: str = "UNKNOWN"  # LOW, MEDIUM, HIGH

    # Métadonnées
    model_version: str = ""
    features_used: List[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'predicted_price': self.predicted_price,
            'predicted_direction': self.predicted_direction,
            'confidence': self.confidence,
            'direction_probabilities': self.direction_probabilities,
            'price_change_pct': self.price_change_pct,
            'volatility_prediction': self.volatility_prediction,
            'current_price': self.current_price,
            'prediction_horizon': self.prediction_horizon,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'model_confidence': self.model_confidence,
            'feature_quality_score': self.feature_quality_score,
            'prediction_reliability': self.prediction_reliability,
            'model_version': self.model_version,
            'features_used': self.features_used or [],
            'processing_time_ms': self.processing_time_ms
        }


@dataclass
class PredictionConfig:
    """Configuration pour les prédictions"""
    # Modèle
    model_path: Optional[str] = None
    auto_load_best_model: bool = True
    model_update_frequency_hours: int = 24

    # Données
    feature_update_frequency_seconds: int = 60
    min_data_quality_score: float = 0.7
    max_data_age_minutes: int = 5

    # Prédictions
    prediction_cache_size: int = 1000
    cache_timeout_seconds: int = 30
    batch_prediction_size: int = 50

    # Qualité
    min_confidence_threshold: float = 0.6
    min_reliability_threshold: str = "MEDIUM"
    enable_prediction_validation: bool = True

    # Performance
    max_prediction_time_ms: float = 100.0
    enable_performance_monitoring: bool = True


class ModelPredictor:
    """Prédicteur utilisant les modèles LSTM pour les prédictions temps réel"""

    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()

        # Modèle et état
        self.model = None
        self.model_info = {}
        self.last_model_update = None
        self.model_lock = Lock()

        # Cache des prédictions
        self.prediction_cache = {}
        self.feature_cache = {}
        self.last_feature_update = None

        # Statistiques
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_processing_time_ms': 0.0,
            'avg_confidence': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_prediction_time': None,
            'model_loads': 0,
            'feature_updates': 0
        }

        # Historique des prédictions
        self.prediction_history = []
        self.max_history_size = 10000

        # Auto-chargement du modèle
        if self.config.auto_load_best_model:
            self._auto_load_best_model()

        logger.info("Prédicteur LSTM initialisé")

    def _auto_load_best_model(self) -> bool:
        """Charge automatiquement le meilleur modèle disponible"""
        try:
            if self.config.model_path and os.path.exists(self.config.model_path):
                return self.load_model(self.config.model_path)

            # Chercher le meilleur modèle dans le dossier des modèles
            models_dir = "models"
            if not os.path.exists(models_dir):
                logger.warning("Dossier des modèles non trouvé")
                return False

            # Lister tous les modèles .h5
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]

            if not model_files:
                logger.warning("Aucun modèle trouvé")
                return False

            # Trier par date de modification (le plus récent en premier)
            model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)

            # Charger le modèle le plus récent
            best_model_path = os.path.join(models_dir, model_files[0])
            logger.info(f"Chargement automatique du modèle: {best_model_path}")

            return self.load_model(best_model_path)

        except Exception as e:
            logger.error(f"Erreur lors du chargement automatique: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Charge un modèle LSTM

        Args:
            model_path: Chemin vers le modèle

        Returns:
            True si chargement réussi
        """
        try:
            with self.model_lock:
                logger.info(f"📥 Chargement du modèle: {model_path}")

                if not os.path.exists(model_path):
                    logger.error(f"Modèle non trouvé: {model_path}")
                    return False

                # Charger le modèle Keras
                keras_model = keras.models.load_model(model_path)

                # Créer l'objet LSTMModel
                self.model = LSTMModel()
                self.model.model = keras_model

                # Charger les métadonnées
                self._load_model_metadata(model_path)

                # Mettre à jour les statistiques
                self.last_model_update = datetime.now(timezone.utc)
                self.prediction_stats['model_loads'] += 1

                logger.info(f"✅ Modèle chargé avec succès")
                return True

        except Exception as e:
            logger.error(f"Erreur chargement du modèle: {e}")
            return False

    def _load_model_metadata(self, model_path: str):
        """Charge les métadonnées du modèle"""
        try:
            # Chercher le fichier d'historique correspondant
            history_path = model_path.replace('.h5', '_history.json')

            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    data = json.load(f)

                    if 'model_config' in data:
                        self.model_info.update(data['model_config'])

                    if 'training_results' in data:
                        training_results = data['training_results']
                        self.model_info['last_accuracy'] = training_results.get('best_metrics', {}).get('accuracy', 0.0)
                        self.model_info['last_training_time'] = training_results.get('training_time', 0.0)

            # Informations basiques du modèle
            if self.model and self.model.model:
                self.model_info['input_shape'] = self.model.model.input_shape
                self.model_info['output_shape'] = self.model.model.output_shape
                self.model_info['parameters'] = self.model.model.count_params()

            logger.debug(f"Métadonnées du modèle chargées: {len(self.model_info)} éléments")

        except Exception as e:
            logger.warning(f"Impossible de charger les métadonnées: {e}")

    def predict(self, symbol: str = "R_10", timeframe: str = "1m",
                force_refresh: bool = False) -> Optional[PredictionResult]:
        """
        Effectue une prédiction pour un symbole

        Args:
            symbol: Symbole à prédire
            timeframe: Timeframe des données
            force_refresh: Forcer le rafraîchissement du cache

        Returns:
            Résultat de prédiction ou None si échec
        """
        start_time = time.time()

        try:
            # Vérifier si le modèle est chargé
            if self.model is None or self.model.model is None:
                logger.warning("Aucun modèle chargé pour les prédictions")
                self._auto_load_best_model()

                if self.model is None:
                    self.prediction_stats['failed_predictions'] += 1
                    return None

            # Vérifier le cache
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now(timezone.utc)

            if not force_refresh and cache_key in self.prediction_cache:
                cached_result, cache_time = self.prediction_cache[cache_key]
                if (current_time - cache_time).total_seconds() < self.config.cache_timeout_seconds:
                    self.prediction_stats['cache_hits'] += 1
                    return cached_result

            self.prediction_stats['cache_misses'] += 1

            # Récupérer et préparer les features
            features = self._get_prediction_features(symbol, timeframe)
            if features is None:
                logger.warning("Impossible de récupérer les features pour la prédiction")
                self.prediction_stats['failed_predictions'] += 1
                return None

            # Effectuer la prédiction
            with self.model_lock:
                prediction_result = self._make_prediction(features, symbol)

            if prediction_result is None:
                self.prediction_stats['failed_predictions'] += 1
                return None

            # Calculer le temps de traitement
            processing_time = (time.time() - start_time) * 1000
            prediction_result.processing_time_ms = processing_time

            # Valider la prédiction
            if self.config.enable_prediction_validation:
                prediction_result = self._validate_prediction(prediction_result)

            # Mettre en cache
            self.prediction_cache[cache_key] = (prediction_result, current_time)

            # Nettoyer le cache si nécessaire
            if len(self.prediction_cache) > self.config.prediction_cache_size:
                self._cleanup_cache()

            # Ajouter à l'historique
            self._add_to_history(prediction_result)

            # Mettre à jour les statistiques
            self._update_prediction_stats(prediction_result, processing_time)

            logger.debug(f"Prédiction effectuée en {processing_time:.1f}ms")
            return prediction_result

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            self.prediction_stats['failed_predictions'] += 1
            return None

    def _get_prediction_features(self, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Récupère et prépare les features pour la prédiction"""
        try:
            # Vérifier le cache des features
            cache_key = f"features_{symbol}_{timeframe}"
            current_time = datetime.now(timezone.utc)

            if (cache_key in self.feature_cache and
                    self.last_feature_update and
                    (
                            current_time - self.last_feature_update).total_seconds() < self.config.feature_update_frequency_seconds):
                return self.feature_cache[cache_key]

            # Récupérer les données récentes
            end_time = current_time
            start_time = end_time - timedelta(hours=2)  # 2 heures de données

            feature_set = prepare_training_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_time,
                end_date=end_time
            )

            if feature_set.price_features.empty:
                logger.warning("Aucune donnée récente disponible")
                return None

            # Préparer les séquences
            X, _ = self.model.prepare_sequences(feature_set)

            if len(X) == 0:
                logger.warning("Impossible de créer des séquences de prédiction")
                return None

            # Prendre la dernière séquence pour la prédiction
            prediction_sequence = X[-1:]  # Garder la dimension batch

            # Mettre en cache
            self.feature_cache[cache_key] = prediction_sequence
            self.last_feature_update = current_time
            self.prediction_stats['feature_updates'] += 1

            return prediction_sequence

        except Exception as e:
            logger.error(f"Erreur récupération des features: {e}")
            return None

    def _make_prediction(self, features: np.ndarray, symbol: str) -> Optional[PredictionResult]:
        """Effectue la prédiction avec le modèle"""
        try:
            # Prédiction avec le modèle
            raw_prediction = self.model.model.predict(features, verbose=0)

            # Récupérer le prix actuel
            current_price = self._get_current_price(symbol)
            if current_price is None:
                logger.warning("Impossible de récupérer le prix actuel")
                current_price = 0.0

            # Traiter selon le type de prédiction
            result = PredictionResult(
                current_price=current_price,
                timestamp=datetime.now(timezone.utc),
                model_version=self.model_info.get('version', 'unknown'),
                features_used=self.model.feature_names[:10] if self.model.feature_names else [],
                prediction_horizon=self.model.config.prediction_horizon if self.model else 5
            )

            if isinstance(raw_prediction, list):
                # Multi-output: [prix, direction, confiance]
                if len(raw_prediction) >= 3:
                    price_pred = raw_prediction[0][0, 0] if raw_prediction[0].size > 0 else current_price
                    direction_probs = raw_prediction[1][0] if raw_prediction[1].size > 0 else [0.33, 0.34, 0.33]
                    confidence_pred = raw_prediction[2][0, 0] if raw_prediction[2].size > 0 else 0.5

                    result.predicted_price = float(price_pred)
                    result.model_confidence = float(confidence_pred)

                    # Traiter les probabilités de direction
                    direction_labels = ['DOWN', 'STABLE', 'UP']
                    result.direction_probabilities = {
                        direction_labels[i]: float(prob) for i, prob in enumerate(direction_probs)
                    }

                    # Déterminer la direction prédite
                    max_prob_idx = np.argmax(direction_probs)
                    result.predicted_direction = direction_labels[max_prob_idx]
                    result.confidence = float(direction_probs[max_prob_idx])

                    # Calculer le changement de prix prédit
                    if current_price > 0:
                        result.price_change_pct = ((result.predicted_price - current_price) / current_price) * 100

            else:
                # Single output
                if len(raw_prediction.shape) > 1 and raw_prediction.shape[1] > 1:
                    # Classification de direction
                    direction_probs = raw_prediction[0]
                    direction_labels = ['DOWN', 'STABLE', 'UP']

                    result.direction_probabilities = {
                        direction_labels[i]: float(prob) for i, prob in enumerate(direction_probs)
                    }

                    max_prob_idx = np.argmax(direction_probs)
                    result.predicted_direction = direction_labels[max_prob_idx]
                    result.confidence = float(direction_probs[max_prob_idx])
                    result.model_confidence = result.confidence

                else:
                    # Prédiction de prix
                    predicted_price = float(raw_prediction[0, 0])
                    result.predicted_price = predicted_price

                    if current_price > 0:
                        price_change = ((predicted_price - current_price) / current_price) * 100
                        result.price_change_pct = price_change

                        # Déterminer la direction basée sur le changement de prix
                        if price_change > 0.1:
                            result.predicted_direction = 'UP'
                            result.confidence = min(0.9, abs(price_change) / 10)
                        elif price_change < -0.1:
                            result.predicted_direction = 'DOWN'
                            result.confidence = min(0.9, abs(price_change) / 10)
                        else:
                            result.predicted_direction = 'STABLE'
                            result.confidence = 0.6

                        result.model_confidence = result.confidence

            # Calculer la fiabilité de la prédiction
            result.prediction_reliability = self._calculate_reliability(result)

            return result

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Récupère le prix actuel d'un symbole"""
        try:
            latest_price = db_manager.get_latest_price(symbol, "1m")
            if latest_price:
                return latest_price.close_price
            return None
        except Exception as e:
            logger.error(f"Erreur récupération prix actuel: {e}")
            return None

    def _calculate_reliability(self, result: PredictionResult) -> str:
        """Calcule la fiabilité de la prédiction"""
        try:
            # Facteurs de fiabilité
            confidence_score = result.confidence
            model_confidence_score = result.model_confidence

            # Score basé sur l'historique du modèle
            model_accuracy = self.model_info.get('last_accuracy', 0.5)

            # Score combiné
            reliability_score = (confidence_score * 0.4 +
                                 model_confidence_score * 0.3 +
                                 model_accuracy * 0.3)

            if reliability_score >= 0.8:
                return "HIGH"
            elif reliability_score >= 0.6:
                return "MEDIUM"
            else:
                return "LOW"

        except Exception as e:
            logger.error(f"Erreur calcul de fiabilité: {e}")
            return "UNKNOWN"

    def _validate_prediction(self, result: PredictionResult) -> PredictionResult:
        """Valide et ajuste la prédiction si nécessaire"""
        try:
            # Vérifier les seuils de confiance
            if result.confidence < self.config.min_confidence_threshold:
                result.prediction_reliability = "LOW"
                logger.debug(f"Prédiction de faible confiance: {result.confidence:.3f}")

            # Vérifier la cohérence des prédictions de prix et de direction
            if result.predicted_price and result.price_change_pct:
                predicted_direction_from_price = 'UP' if result.price_change_pct > 0.1 else (
                    'DOWN' if result.price_change_pct < -0.1 else 'STABLE')

                if result.predicted_direction != predicted_direction_from_price:
                    logger.debug("Incohérence entre prédiction de prix et de direction")
                    # Réduire la confiance en cas d'incohérence
                    result.confidence *= 0.8
                    result.prediction_reliability = "LOW"

            # Vérifier les limites de prix (pour Volatility 10, changements extrêmes improbables)
            if result.price_change_pct and abs(result.price_change_pct) > 5.0:  # Plus de 5% de changement
                logger.warning(f"Prédiction de changement de prix extrême: {result.price_change_pct:.2f}%")
                result.confidence *= 0.5
                result.prediction_reliability = "LOW"

            return result

        except Exception as e:
            logger.error(f"Erreur validation de prédiction: {e}")
            return result

    def _cleanup_cache(self):
        """Nettoie les anciens éléments du cache"""
        try:
            # Garder seulement les éléments les plus récents
            if len(self.prediction_cache) > self.config.prediction_cache_size:
                sorted_items = sorted(
                    self.prediction_cache.items(),
                    key=lambda x: x[1][1],  # Trier par timestamp
                    reverse=True
                )

                # Garder seulement les plus récents
                keep_items = sorted_items[:self.config.prediction_cache_size // 2]
                self.prediction_cache = dict(keep_items)

                logger.debug(f"Cache nettoyé: {len(keep_items)} éléments conservés")

        except Exception as e:
            logger.error(f"Erreur nettoyage du cache: {e}")

    def _add_to_history(self, result: PredictionResult):
        """Ajoute une prédiction à l'historique"""
        try:
            self.prediction_history.append(result)

            # Limiter la taille de l'historique
            if len(self.prediction_history) > self.max_history_size:
                self.prediction_history = self.prediction_history[-self.max_history_size // 2:]

        except Exception as e:
            logger.error(f"Erreur ajout à l'historique: {e}")

    def _update_prediction_stats(self, result: PredictionResult, processing_time: float):
        """Met à jour les statistiques de prédiction"""
        try:
            self.prediction_stats['total_predictions'] += 1
            self.prediction_stats['successful_predictions'] += 1
            self.prediction_stats['last_prediction_time'] = result.timestamp

            # Temps de traitement moyen
            total_predictions = self.prediction_stats['total_predictions']
            old_avg = self.prediction_stats['avg_processing_time_ms']
            self.prediction_stats['avg_processing_time_ms'] = (
                    (old_avg * (total_predictions - 1) + processing_time) / total_predictions
            )

            # Confiance moyenne
            old_avg_conf = self.prediction_stats['avg_confidence']
            self.prediction_stats['avg_confidence'] = (
                    (old_avg_conf * (total_predictions - 1) + result.confidence) / total_predictions
            )

        except Exception as e:
            logger.error(f"Erreur mise à jour des statistiques: {e}")

    def batch_predict(self, symbols: List[str], timeframe: str = "1m") -> Dict[str, Optional[PredictionResult]]:
        """
        Effectue des prédictions en lot pour plusieurs symboles

        Args:
            symbols: Liste des symboles
            timeframe: Timeframe des données

        Returns:
            Dictionnaire des prédictions par symbole
        """
        try:
            logger.info(f"Prédictions en lot pour {len(symbols)} symboles")

            results = {}

            for symbol in symbols:
                try:
                    result = self.predict(symbol, timeframe)
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Erreur prédiction pour {symbol}: {e}")
                    results[symbol] = None

            successful_predictions = sum(1 for r in results.values() if r is not None)
            logger.info(f"Prédictions en lot terminées: {successful_predictions}/{len(symbols)} réussies")

            return results

        except Exception as e:
            logger.error(f"Erreur prédictions en lot: {e}")
            return {}

    def get_prediction_confidence_distribution(self, hours: int = 24) -> Dict[str, int]:
        """Retourne la distribution des niveaux de confiance sur une période"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            recent_predictions = [
                p for p in self.prediction_history
                if p.timestamp and p.timestamp > cutoff_time
            ]

            distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}

            for prediction in recent_predictions:
                reliability = prediction.prediction_reliability
                if reliability in distribution:
                    distribution[reliability] += 1
                else:
                    distribution["UNKNOWN"] += 1

            return distribution

        except Exception as e:
            logger.error(f"Erreur calcul distribution de confiance: {e}")
            return {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}

    def get_prediction_accuracy(self, hours: int = 24) -> Dict[str, float]:
        """
        Calcule la précision des prédictions récentes (nécessite des données de validation)

        Args:
            hours: Nombre d'heures à analyser

        Returns:
            Dictionnaire avec les métriques de précision
        """
        try:
            # TODO: Implémenter la validation des prédictions avec les données réelles
            # Cela nécessiterait de stocker les prédictions et de les comparer avec les prix réels

            return {
                'directional_accuracy': 0.0,
                'price_accuracy_mae': 0.0,
                'confidence_accuracy': 0.0,
                'predictions_validated': 0
            }

        except Exception as e:
            logger.error(f"Erreur calcul de précision: {e}")
            return {'error': str(e)}

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de prédiction"""
        stats = self.prediction_stats.copy()

        # Ajouter des informations supplémentaires
        stats['model_loaded'] = self.model is not None
        stats['model_info'] = self.model_info.copy()
        stats['cache_size'] = len(self.prediction_cache)
        stats['history_size'] = len(self.prediction_history)
        stats['last_model_update'] = self.last_model_update.isoformat() if self.last_model_update else None

        # Calculer le taux de réussite
        total = stats['total_predictions']
        if total > 0:
            stats['success_rate'] = stats['successful_predictions'] / total
            stats['failure_rate'] = stats['failed_predictions'] / total
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])

        return stats


# Instance globale
model_predictor = ModelPredictor()


# Fonctions utilitaires
def predict_price(symbol: str = "R_10", timeframe: str = "1m") -> Optional[PredictionResult]:
    """Fonction utilitaire pour prédire le prix d'un symbole"""
    return model_predictor.predict(symbol, timeframe)


def get_prediction_stats() -> Dict[str, Any]:
    """Fonction utilitaire pour récupérer les statistiques de prédiction"""
    return model_predictor.get_prediction_stats()


if __name__ == "__main__":
    # Test du prédicteur
    print("🔮 Test du prédicteur LSTM...")

    try:
        # Configuration de test
        test_config = PredictionConfig(
            auto_load_best_model=True,
            min_confidence_threshold=0.3,  # Seuil bas pour les tests
            cache_timeout_seconds=10
        )

        predictor = ModelPredictor(test_config)

        print(f"✅ Prédicteur configuré")
        print(f"   Modèle chargé: {predictor.model is not None}")
        print(f"   Auto-chargement: {test_config.auto_load_best_model}")

        # Test de prédiction (si modèle disponible)
        if predictor.model:
            try:
                result = predictor.predict("R_10", "1m")
                if result:
                    print(f"\n🔮 Prédiction de test:")
                    print(
                        f"   Prix prédit: {result.predicted_price:.5f}" if result.predicted_price else "   Prix prédit: N/A")
                    print(f"   Direction: {result.predicted_direction}")
                    print(f"   Confiance: {result.confidence:.3f}")
                    print(f"   Fiabilité: {result.prediction_reliability}")
                    print(f"   Temps de traitement: {result.processing_time_ms:.1f}ms")
                else:
                    print(f"❌ Prédiction échouée")
            except Exception as e:
                print(f"⚠️ Test de prédiction échoué (normal si pas de données): {e}")
        else:
            print(f"⚠️ Aucun modèle chargé - test de prédiction ignoré")

        # Test de prédictions en lot
        try:
            batch_results = predictor.batch_predict(["R_10"])
            print(f"\n📊 Prédictions en lot: {len(batch_results)} résultats")
        except Exception as e:
            print(f"⚠️ Test batch échoué: {e}")

        # Statistiques
        stats = predictor.get_prediction_stats()
        print(f"\n📈 Statistiques:")
        for key, value in list(stats.items())[:8]:  # Afficher les 8 premières
            print(f"   {key}: {value}")

        print("✅ Test du prédicteur réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test du prédicteur échoué: {e}")