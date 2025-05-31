"""
Module d'intelligence artificielle pour le Trading Bot Volatility 10
Centralise les modèles LSTM, l'entraînement et les prédictions

Usage:
    from ai_model import train_model, predict_price, get_model_performance
    from ai_model import LSTMModel, ModelTrainer, ModelPredictor
    from ai_model import ModelConfig, TrainingConfig, PredictionConfig
"""

# Imports des classes principales
from .lstm_model import (
    LSTMModel,
    ModelConfig,
    ModelMetrics,
    ModelArchitecture,
    PredictionType,
    AttentionLayer,
    lstm_model
)

from .trainer import (
    ModelTrainer,
    TrainingConfig,
    TrainingResults,
    model_trainer,
    train_lstm_model
)

from .predictor import (
    ModelPredictor,
    PredictionConfig,
    PredictionResult,
    model_predictor,
    predict_price,
    get_prediction_stats
)

import logging
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import warnings

from config import config
from data import prepare_training_data

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Version du module
__version__ = "1.0.0"

# Informations sur le module
__author__ = "Trading Bot Team"
__description__ = "Module d'intelligence artificielle LSTM pour le Trading Bot Volatility 10"

# Exports principaux
__all__ = [
    # Classes LSTM
    "LSTMModel",
    "ModelConfig",
    "ModelMetrics",
    "ModelArchitecture",
    "PredictionType",
    "AttentionLayer",

    # Classes d'entraînement
    "ModelTrainer",
    "TrainingConfig",
    "TrainingResults",

    # Classes de prédiction
    "ModelPredictor",
    "PredictionConfig",
    "PredictionResult",

    # Instances globales
    "lstm_model",
    "model_trainer",
    "model_predictor",

    # Fonctions utilitaires - Entraînement
    "train_lstm_model",
    "train_and_evaluate_model",
    "optimize_hyperparameters",

    # Fonctions utilitaires - Prédiction
    "predict_price",
    "predict_direction",
    "batch_predict",
    "get_prediction_stats",

    # Fonctions du module
    "get_model_performance",
    "get_ai_pipeline_status",
    "auto_retrain_model",
    "load_best_model",
    "get_ai_insights",
    "validate_model_performance",

    # Métadonnées
    "__version__",
    "__author__",
    "__description__"
]


@dataclass
class ModelPerformance:
    """Performance globale des modèles IA"""
    # Métriques principales
    accuracy: float = 0.0
    directional_accuracy: float = 0.0
    mae: float = 0.0
    sharpe_ratio: float = 0.0

    # Fiabilité
    prediction_consistency: float = 0.0
    confidence_calibration: float = 0.0
    model_stability: float = 0.0

    # Historique
    total_predictions: int = 0
    successful_predictions: int = 0
    last_evaluation: Optional[datetime] = None

    # Comparaison
    benchmark_accuracy: float = 0.5  # Précision aléatoire
    performance_vs_benchmark: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIInsights:
    """Insights générés par l'IA"""
    # Prédictions de marché
    market_direction_confidence: float = 0.0
    volatility_forecast: str = "NORMAL"  # LOW, NORMAL, HIGH
    trend_strength: float = 0.0

    # Patterns détectés
    recurring_patterns: List[str] = None
    anomaly_detection: List[str] = None
    support_resistance_ai: List[float] = None

    # Recommandations
    trading_recommendation: str = "HOLD"  # BUY, SELL, HOLD
    optimal_timeframe: str = "1m"
    risk_assessment: str = "MEDIUM"  # LOW, MEDIUM, HIGH

    # Métadonnées
    analysis_timestamp: datetime = None
    model_versions_used: List[str] = None
    data_quality_score: float = 0.0

    def __post_init__(self):
        if self.recurring_patterns is None:
            self.recurring_patterns = []
        if self.anomaly_detection is None:
            self.anomaly_detection = []
        if self.support_resistance_ai is None:
            self.support_resistance_ai = []
        if self.model_versions_used is None:
            self.model_versions_used = []
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now(timezone.utc)


def train_and_evaluate_model(model_config: ModelConfig = None,
                             training_config: TrainingConfig = None,
                             symbol: str = "R_10") -> Dict[str, Any]:
    """
    Entraîne et évalue un modèle LSTM complet

    Args:
        model_config: Configuration du modèle
        training_config: Configuration de l'entraînement  
        symbol: Symbole à entraîner

    Returns:
        Résultats complets d'entraînement et d'évaluation
    """
    try:
        logger.info(f"🚀 Entraînement et évaluation complète pour {symbol}")

        # Configuration par défaut
        if model_config is None:
            model_config = ModelConfig()
        if training_config is None:
            training_config = TrainingConfig(symbol=symbol)

        # Créer l'entraîneur
        trainer = ModelTrainer(training_config)

        # Entraînement
        training_results = trainer.train_model(model_config)

        # Validation croisée
        cv_results = trainer.cross_validate(model_config, n_splits=3)

        # Évaluation finale
        model_performance = evaluate_model_performance(trainer.model)

        # Compilation des résultats
        complete_results = {
            'training_results': training_results.to_dict(),
            'cross_validation': cv_results,
            'model_performance': model_performance.to_dict(),
            'model_config': asdict(model_config),
            'training_config': asdict(training_config),
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"✅ Entraînement terminé - Précision: {model_performance.accuracy:.3f}")
        return complete_results

    except Exception as e:
        logger.error(f"Erreur entraînement et évaluation: {e}")
        return {
            'error': str(e),
            'success': False,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def evaluate_model_performance(model: LSTMModel) -> ModelPerformance:
    """
    Évalue la performance d'un modèle entraîné

    Args:
        model: Modèle LSTM à évaluer

    Returns:
        Métriques de performance
    """
    try:
        performance = ModelPerformance()

        if model is None or model.model is None:
            logger.warning("Aucun modèle à évaluer")
            return performance

        # Récupérer les métriques du modèle
        if hasattr(model, 'metrics'):
            metrics = model.metrics
            performance.accuracy = metrics.accuracy
            performance.directional_accuracy = metrics.directional_accuracy
            performance.mae = metrics.mae
            performance.sharpe_ratio = metrics.sharpe_ratio

        # Calculer les métriques supplémentaires
        performance.last_evaluation = datetime.now(timezone.utc)

        # Performance vs benchmark
        performance.performance_vs_benchmark = performance.accuracy - performance.benchmark_accuracy

        # Évaluation qualitative
        if performance.accuracy > 0.7:
            performance.model_stability = 0.9
        elif performance.accuracy > 0.6:
            performance.model_stability = 0.7
        else:
            performance.model_stability = 0.5

        logger.debug(f"Performance évaluée: accuracy={performance.accuracy:.3f}")
        return performance

    except Exception as e:
        logger.error(f"Erreur évaluation performance: {e}")
        return ModelPerformance()


def optimize_hyperparameters(symbol: str = "R_10",
                             max_trials: int = 20) -> Dict[str, Any]:
    """
    Optimise les hyperparamètres du modèle

    Args:
        symbol: Symbole pour l'optimisation
        max_trials: Nombre maximum d'essais

    Returns:
        Meilleurs hyperparamètres trouvés
    """
    try:
        logger.info(f"🔧 Optimisation des hyperparamètres pour {symbol}")

        best_config = None
        best_score = 0.0
        optimization_results = []

        # Définir l'espace de recherche
        search_space = {
            'lstm_units': [
                [32, 16], [64, 32], [64, 32, 16], [128, 64, 32]
            ],
            'dense_units': [
                [16], [32, 16], [64, 32], [32, 16, 8]
            ],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64]
        }

        # Recherche aléatoire
        import random
        for trial in range(max_trials):
            logger.info(f"📊 Essai {trial + 1}/{max_trials}")

            # Générer une configuration aléatoire
            trial_config = ModelConfig(
                lstm_units=random.choice(search_space['lstm_units']),
                dense_units=random.choice(search_space['dense_units']),
                dropout_rate=random.choice(search_space['dropout_rate']),
                learning_rate=random.choice(search_space['learning_rate']),
                batch_size=random.choice(search_space['batch_size']),
                epochs=20  # Moins d'époques pour l'optimisation
            )

            try:
                # Entraîner avec cette configuration
                trainer = ModelTrainer(TrainingConfig(
                    symbol=symbol,
                    training_days=14,  # Moins de données pour être plus rapide
                    epochs=20
                ))

                results = trainer.train_model(trial_config)
                score = results.best_metrics.accuracy

                optimization_results.append({
                    'trial': trial + 1,
                    'config': asdict(trial_config),
                    'score': score,
                    'metrics': asdict(results.best_metrics)
                })

                if score > best_score:
                    best_score = score
                    best_config = trial_config
                    logger.info(f"✨ Nouveau meilleur score: {score:.3f}")

            except Exception as e:
                logger.warning(f"Essai {trial + 1} échoué: {e}")
                continue

        # Résultats de l'optimisation
        optimization_summary = {
            'best_config': asdict(best_config) if best_config else None,
            'best_score': best_score,
            'trials_completed': len(optimization_results),
            'optimization_results': optimization_results,
            'optimization_timestamp': datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"🎯 Optimisation terminée - Meilleur score: {best_score:.3f}")
        return optimization_summary

    except Exception as e:
        logger.error(f"Erreur optimisation hyperparamètres: {e}")
        return {
            'error': str(e),
            'best_score': 0.0,
            'trials_completed': 0
        }


def predict_direction(symbol: str = "R_10", timeframe: str = "1m") -> Optional[str]:
    """
    Prédit uniquement la direction du mouvement

    Args:
        symbol: Symbole à prédire
        timeframe: Timeframe des données

    Returns:
        Direction prédite ('UP', 'DOWN', 'STABLE') ou None
    """
    try:
        result = model_predictor.predict(symbol, timeframe)
        return result.predicted_direction if result else None

    except Exception as e:
        logger.error(f"Erreur prédiction direction: {e}")
        return None


def batch_predict(symbols: List[str], timeframe: str = "1m") -> Dict[str, Optional[PredictionResult]]:
    """
    Prédit pour plusieurs symboles en une fois

    Args:
        symbols: Liste des symboles
        timeframe: Timeframe des données

    Returns:
        Dictionnaire des prédictions
    """
    return model_predictor.batch_predict(symbols, timeframe)


def get_model_performance() -> ModelPerformance:
    """
    Retourne la performance actuelle des modèles

    Returns:
        Métriques de performance consolidées
    """
    try:
        # Performance du modèle actuel
        if model_predictor.model:
            model_perf = evaluate_model_performance(model_predictor.model)
        else:
            model_perf = ModelPerformance()

        # Ajouter les statistiques de prédiction
        pred_stats = model_predictor.get_prediction_stats()

        model_perf.total_predictions = pred_stats.get('total_predictions', 0)
        model_perf.successful_predictions = pred_stats.get('successful_predictions', 0)

        if model_perf.total_predictions > 0:
            model_perf.prediction_consistency = (
                    model_perf.successful_predictions / model_perf.total_predictions
            )

        return model_perf

    except Exception as e:
        logger.error(f"Erreur récupération performance: {e}")
        return ModelPerformance()


def get_ai_pipeline_status() -> Dict[str, Any]:
    """
    Retourne le statut complet du pipeline IA

    Returns:
        Statut de tous les composants IA
    """
    try:
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_health': 'unknown',
            'components': {}
        }

        # Statut du modèle LSTM
        model_loaded = lstm_model.model is not None
        status['components']['lstm_model'] = {
            'loaded': model_loaded,
            'architecture': lstm_model.config.architecture.value if lstm_model else 'unknown',
            'info': lstm_model.get_model_info() if model_loaded else {}
        }

        # Statut de l'entraîneur
        trainer_stats = model_trainer.get_training_stats()
        status['components']['trainer'] = {
            'models_trained': trainer_stats.get('models_trained', 0),
            'last_training': trainer_stats.get('last_training_date'),
            'best_accuracy': trainer_stats.get('best_accuracy_achieved', 0.0)
        }

        # Statut du prédicteur
        predictor_stats = model_predictor.get_prediction_stats()
        status['components']['predictor'] = {
            'model_loaded': predictor_stats.get('model_loaded', False),
            'total_predictions': predictor_stats.get('total_predictions', 0),
            'success_rate': predictor_stats.get('success_rate', 0.0),
            'avg_processing_time': predictor_stats.get('avg_processing_time_ms', 0.0)
        }

        # Santé globale
        health_factors = [
            model_loaded,
            trainer_stats.get('models_trained', 0) > 0,
            predictor_stats.get('success_rate', 0.0) > 0.7
        ]

        if all(health_factors):
            status['overall_health'] = 'excellent'
        elif any(health_factors):
            status['overall_health'] = 'partial'
        else:
            status['overall_health'] = 'critical'

        return status

    except Exception as e:
        logger.error(f"Erreur récupération statut pipeline: {e}")
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_health': 'error',
            'error': str(e)
        }


def auto_retrain_model(symbol: str = "R_10",
                       performance_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Redéclenche automatiquement l'entraînement si les performances dégradent

    Args:
        symbol: Symbole pour le réentraînement
        performance_threshold: Seuil de dégradation pour déclencher le réentraînement

    Returns:
        Résultats du réentraînement
    """
    try:
        logger.info(f"🔄 Vérification de la nécessité de réentraîner pour {symbol}")

        # Évaluer les performances actuelles
        current_performance = get_model_performance()

        # Vérifier si un réentraînement est nécessaire
        needs_retraining = False
        reasons = []

        # Performance dégradée
        if current_performance.accuracy < (current_performance.benchmark_accuracy + performance_threshold):
            needs_retraining = True
            reasons.append(f"Performance dégradée: {current_performance.accuracy:.3f}")

        # Modèle trop ancien
        if model_predictor.last_model_update:
            age_hours = (datetime.now(timezone.utc) - model_predictor.last_model_update).total_seconds() / 3600
            if age_hours > config.ai_model.model_retrain_frequency_hours:
                needs_retraining = True
                reasons.append(f"Modèle ancien: {age_hours:.1f} heures")

        # Taux d'échec trop élevé
        pred_stats = model_predictor.get_prediction_stats()
        failure_rate = pred_stats.get('failure_rate', 0.0)
        if failure_rate > 0.2:  # Plus de 20% d'échecs
            needs_retraining = True
            reasons.append(f"Taux d'échec élevé: {failure_rate:.1%}")

        if not needs_retraining:
            return {
                'retraining_triggered': False,
                'reason': 'Performance satisfaisante',
                'current_accuracy': current_performance.accuracy,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        logger.info(f"🚀 Réentraînement nécessaire: {', '.join(reasons)}")

        # Lancer le réentraînement
        retraining_results = train_and_evaluate_model(symbol=symbol)

        # Charger le nouveau modèle dans le prédicteur
        if 'training_results' in retraining_results:
            model_path = retraining_results['training_results'].get('model_path')
            if model_path and os.path.exists(model_path):
                model_predictor.load_model(model_path)
                logger.info("✅ Nouveau modèle chargé dans le prédicteur")

        return {
            'retraining_triggered': True,
            'reasons': reasons,
            'retraining_results': retraining_results,
            'new_performance': get_model_performance().to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Erreur réentraînement automatique: {e}")
        return {
            'retraining_triggered': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def load_best_model(models_dir: str = "models") -> bool:
    """
    Charge le meilleur modèle disponible

    Args:
        models_dir: Dossier contenant les modèles

    Returns:
        True si chargement réussi
    """
    try:
        if not os.path.exists(models_dir):
            logger.warning(f"Dossier des modèles non trouvé: {models_dir}")
            return False

        # Lister tous les fichiers d'historique (contiennent les métriques)
        history_files = [f for f in os.listdir(models_dir) if f.endswith('_history.json')]

        if not history_files:
            logger.warning("Aucun fichier d'historique trouvé")
            return False

        best_model_path = None
        best_accuracy = 0.0

        # Parcourir tous les modèles pour trouver le meilleur
        for history_file in history_files:
            try:
                history_path = os.path.join(models_dir, history_file)

                with open(history_path, 'r') as f:
                    data = json.load(f)

                # Récupérer la précision
                training_results = data.get('training_results', {})
                best_metrics = training_results.get('best_metrics', {})
                accuracy = best_metrics.get('accuracy', 0.0)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # Construire le chemin du modèle correspondant
                    model_filename = history_file.replace('_history.json', '.h5')
                    model_path = os.path.join(models_dir, model_filename)

                    if os.path.exists(model_path):
                        best_model_path = model_path

            except Exception as e:
                logger.warning(f"Erreur lecture de {history_file}: {e}")
                continue

        if best_model_path:
            success = model_predictor.load_model(best_model_path)
            if success:
                logger.info(f"✅ Meilleur modèle chargé: {best_model_path} (accuracy: {best_accuracy:.3f})")
            return success
        else:
            logger.warning("Aucun modèle valide trouvé")
            return False

    except Exception as e:
        logger.error(f"Erreur chargement du meilleur modèle: {e}")
        return False


def get_ai_insights(symbol: str = "R_10", timeframe: str = "1m") -> AIInsights:
    """
    Génère des insights IA avancés sur le marché

    Args:
        symbol: Symbole à analyser
        timeframe: Timeframe des données

    Returns:
        Insights générés par l'IA
    """
    try:
        insights = AIInsights()

        # Prédiction principale
        prediction = model_predictor.predict(symbol, timeframe)

        if prediction:
            insights.market_direction_confidence = prediction.confidence
            insights.trading_recommendation = prediction.predicted_direction

            # Évaluation du risque basée sur la confiance
            if prediction.confidence > 0.8:
                insights.risk_assessment = "LOW"
            elif prediction.confidence > 0.6:
                insights.risk_assessment = "MEDIUM"
            else:
                insights.risk_assessment = "HIGH"

            # Informations sur le modèle
            insights.model_versions_used = [prediction.model_version]

        # Prédiction de volatilité (basée sur les patterns récents)
        try:
            # Récupérer les données récentes pour analyser la volatilité
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=2)

            feature_set = prepare_training_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_time,
                end_date=end_time
            )

            if not feature_set.price_features.empty:
                # Calculer la volatilité récente
                prices = feature_set.price_features.get('close', pd.Series())
                if len(prices) > 10:
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()

                    if volatility > 0.01:  # Plus de 1%
                        insights.volatility_forecast = "HIGH"
                    elif volatility < 0.005:  # Moins de 0.5%
                        insights.volatility_forecast = "LOW"
                    else:
                        insights.volatility_forecast = "NORMAL"

                    # Force de tendance
                    if len(prices) > 20:
                        sma_short = prices.rolling(10).mean().iloc[-1]
                        sma_long = prices.rolling(20).mean().iloc[-1]
                        current_price = prices.iloc[-1]

                        trend_score = 0.0
                        if current_price > sma_short > sma_long:
                            trend_score = (current_price - sma_long) / sma_long
                        elif current_price < sma_short < sma_long:
                            trend_score = (sma_long - current_price) / sma_long

                        insights.trend_strength = min(1.0, abs(trend_score) * 100)

        except Exception as e:
            logger.warning(f"Erreur analyse de volatilité: {e}")

        # Timeframe optimal (basé sur la performance du modèle)
        performance = get_model_performance()
        if performance.accuracy > 0.7:
            insights.optimal_timeframe = timeframe
        else:
            # Suggérer un timeframe différent si performance faible
            timeframe_alternatives = {"1m": "5m", "5m": "15m", "15m": "1h"}
            insights.optimal_timeframe = timeframe_alternatives.get(timeframe, timeframe)

        # Score de qualité des données
        pred_stats = model_predictor.get_prediction_stats()
        insights.data_quality_score = pred_stats.get('success_rate', 0.5)

        return insights

    except Exception as e:
        logger.error(f"Erreur génération insights IA: {e}")
        return AIInsights()


def validate_model_performance(min_accuracy: float = 0.55,
                               min_predictions: int = 100) -> Dict[str, Any]:
    """
    Valide que les performances du modèle respectent les critères

    Args:
        min_accuracy: Précision minimale requise
        min_predictions: Nombre minimum de prédictions pour valider

    Returns:
        Résultats de validation
    """
    try:
        performance = get_model_performance()
        pred_stats = model_predictor.get_prediction_stats()

        validation_results = {
            'validation_passed': True,
            'issues': [],
            'recommendations': [],
            'performance_metrics': performance.to_dict(),
            'validation_timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Vérifier la précision
        if performance.accuracy < min_accuracy:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(
                f"Précision insuffisante: {performance.accuracy:.3f} < {min_accuracy:.3f}"
            )
            validation_results['recommendations'].append("Réentraîner le modèle avec plus de données")

        # Vérifier le nombre de prédictions
        total_predictions = pred_stats.get('total_predictions', 0)
        if total_predictions < min_predictions:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(
                f"Pas assez de prédictions pour valider: {total_predictions} < {min_predictions}"
            )
            validation_results['recommendations'].append("Attendre plus de prédictions avant validation")

        # Vérifier le taux de succès
        success_rate = pred_stats.get('success_rate', 0.0)
        if success_rate < 0.8:  # Moins de 80% de prédictions réussies
            validation_results['issues'].append(
                f"Taux de succès faible: {success_rate:.1%}"
            )
            validation_results['recommendations'].append("Vérifier la qualité des données d'entrée")

        # Vérifier les temps de traitement
        avg_time = pred_stats.get('avg_processing_time_ms', 0.0)
        if avg_time > 500:  # Plus de 500ms en moyenne
            validation_results['issues'].append(
                f"Temps de traitement élevé: {avg_time:.1f}ms"
            )
            validation_results['recommendations'].append("Optimiser l'architecture du modèle")

        # Recommandations générales
        if validation_results['validation_passed']:
            validation_results['recommendations'].append("Modèle validé - Performance satisfaisante")
        else:
            validation_results['recommendations'].append("Modèle nécessite une attention immédiate")

        return validation_results

    except Exception as e:
        logger.error(f"Erreur validation performance: {e}")
        return {
            'validation_passed': False,
            'error': str(e),
            'validation_timestamp': datetime.now(timezone.utc).isoformat()
        }


def print_ai_module_banner():
    """Affiche la bannière du module IA"""
    print("=" * 80)
    print("🧠 TRADING BOT VOLATILITY 10 - MODULE D'INTELLIGENCE ARTIFICIELLE")
    print("=" * 80)
    print(f"📦 Version: {__version__}")
    print(f"👥 Auteur: {__author__}")
    print(f"📝 Description: {__description__}")

    # Statut du pipeline IA
    status = get_ai_pipeline_status()
    health_emoji = {
        "excellent": "✅",
        "partial": "⚠️",
        "critical": "❌",
        "error": "💥",
        "unknown": "❓"
    }

    print(
        f"\n🏥 Santé du pipeline IA: {health_emoji.get(status['overall_health'], '❓')} {status['overall_health'].upper()}")

    # Composants
    components = status.get('components', {})
    for component, info in components.items():
        if component == 'lstm_model':
            loaded = info.get('loaded', False)
            arch = info.get('architecture', 'unknown')
            print(f"   {'✅' if loaded else '❌'} Modèle LSTM: {arch}")
        elif component == 'trainer':
            models_trained = info.get('models_trained', 0)
            print(f"   📚 Entraîneur: {models_trained} modèles entraînés")
        elif component == 'predictor':
            model_loaded = info.get('model_loaded', False)
            success_rate = info.get('success_rate', 0.0)
            print(f"   {'✅' if model_loaded else '❌'} Prédicteur: {success_rate:.1%} succès")

    # Performance
    performance = get_model_performance()
    print(f"\n📊 Performance:")
    print(f"   Précision: {performance.accuracy:.1%}")
    print(f"   Précision directionnelle: {performance.directional_accuracy:.1%}")
    print(f"   Prédictions totales: {performance.total_predictions}")

    print("=" * 80)


# Initialisation automatique au chargement du module
try:
    logger.info(f"Module IA chargé (version {__version__})")

    # Vérifier que tous les sous-modules sont chargés
    components_loaded = [
        lstm_model is not None,
        model_trainer is not None,
        model_predictor is not None
    ]

    if all(components_loaded):
        logger.info("✅ Tous les modules IA sont opérationnels")

        # Tentative de chargement automatique du meilleur modèle
        if model_predictor.config.auto_load_best_model:
            load_best_model()

    else:
        logger.warning("⚠️ Certains modules IA ne sont pas chargés")

except Exception as e:
    logger.error(f"⚠️ Erreur lors de l'initialisation du module IA: {e}")
    # Ne pas faire planter l'import, permettre l'utilisation partielle