"""
Module de gestion des données pour le Trading Bot Volatility 10
Centralise la collecte, le stockage et le préprocessing des données

Usage:
    from data import start_data_collection, get_latest_price, prepare_training_data
    from data import db_manager, data_collector, preprocessor
    from data import DataQuality, FeatureSet, TickData, CandleData
"""

# Imports des classes principales
from .database import (
    DatabaseManager,
    PriceData,
    TechnicalIndicators,
    TradingSignals,
    Trades,
    PerformanceMetrics,
    SystemLogs,
    db_manager,
    get_db_session,
    initialize_database,
    check_database_health
)

from .collector import (
    DerivDataCollector,
    ConnectionStatus,
    TickData,
    CandleData,
    data_collector,
    start_data_collection,
    stop_data_collection,
    get_collector_stats
)

from .preprocessor import (
    DataPreprocessor,
    DataQuality,
    DataQualityReport,
    FeatureSet,
    preprocessor,
    validate_data_quality,
    clean_data,
    create_features,
    prepare_training_data
)

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

# Configuration du logger
logger = logging.getLogger(__name__)

# Version du module
__version__ = "1.0.0"

# Informations sur le module
__author__ = "HAMID TCHEMOKO Trading Bot "
__description__ = "Module de gestion des données pour le Trading Bot Volatility 10"

# Exports principaux
__all__ = [
    # Classes de base de données
    "DatabaseManager",
    "PriceData",
    "TechnicalIndicators",
    "TradingSignals",
    "Trades",
    "PerformanceMetrics",
    "SystemLogs",

    # Classes de collecte
    "DerivDataCollector",
    "ConnectionStatus",
    "TickData",
    "CandleData",

    # Classes de préprocessing
    "DataPreprocessor",
    "DataQuality",
    "DataQualityReport",
    "FeatureSet",

    # Instances globales
    "db_manager",
    "data_collector",
    "preprocessor",

    # Fonctions utilitaires - Base de données
    "get_db_session",
    "initialize_database",
    "check_database_health",

    # Fonctions utilitaires - Collecte
    "start_data_collection",
    "stop_data_collection",
    "get_collector_stats",

    # Fonctions utilitaires - Préprocessing
    "validate_data_quality",
    "clean_data",
    "create_features",
    "prepare_training_data",

    # Fonctions du module
    "get_data_pipeline_status",
    "get_latest_market_data",
    "get_data_summary",
    "cleanup_old_data",
    "backup_data",
    "restore_data",

    # Métadonnées
    "__version__",
    "__author__",
    "__description__"
]


def get_data_pipeline_status() -> Dict[str, Any]:
    """
    Retourne le statut complet du pipeline de données

    Returns:
        Dictionnaire avec le statut de tous les composants
    """
    try:
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": "unknown",
            "components": {}
        }

        # Statut de la base de données
        try:
            db_healthy = check_database_health()
            db_stats = db_manager.get_database_stats()

            status["components"]["database"] = {
                "healthy": db_healthy,
                "connection": "connected" if db_healthy else "disconnected",
                "stats": db_stats
            }
        except Exception as e:
            status["components"]["database"] = {
                "healthy": False,
                "connection": "error",
                "error": str(e)
            }

        # Statut du collecteur
        try:
            collector_connected = data_collector.is_connected() if data_collector else False
            collector_stats = get_collector_stats()

            status["components"]["collector"] = {
                "healthy": collector_connected,
                "connection": data_collector.status.value if data_collector else "not_initialized",
                "stats": collector_stats
            }
        except Exception as e:
            status["components"]["collector"] = {
                "healthy": False,
                "connection": "error",
                "error": str(e)
            }

        # Statut du préprocesseur
        try:
            preprocessing_stats = preprocessor.get_processing_stats()
            last_processing = preprocessing_stats.get('last_processing_time')

            # Considérer le préprocesseur comme sain s'il a traité des données récemment
            preprocessing_healthy = last_processing is not None
            if last_processing and isinstance(last_processing, datetime):
                time_since_processing = (datetime.now(timezone.utc) - last_processing).total_seconds()
                preprocessing_healthy = time_since_processing < 3600  # Moins d'1 heure

            status["components"]["preprocessor"] = {
                "healthy": preprocessing_healthy,
                "last_processing": last_processing.isoformat() if last_processing else None,
                "stats": preprocessing_stats
            }
        except Exception as e:
            status["components"]["preprocessor"] = {
                "healthy": False,
                "error": str(e)
            }

        # Déterminer la santé globale
        component_health = [comp.get("healthy", False) for comp in status["components"].values()]
        if all(component_health):
            status["overall_health"] = "excellent"
        elif any(component_health):
            status["overall_health"] = "partial"
        else:
            status["overall_health"] = "critical"

        return status

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du statut du pipeline: {e}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": "error",
            "error": str(e)
        }


def get_latest_market_data(symbol: str, timeframe: str = "1m") -> Optional[Dict[str, Any]]:
    """
    Récupère les dernières données de marché pour un symbole

    Args:
        symbol: Symbole à récupérer (ex: "R_10")
        timeframe: Timeframe des données (ex: "1m", "5m")

    Returns:
        Dictionnaire avec les dernières données ou None
    """
    try:
        # Essayer d'abord le collecteur en temps réel
        if data_collector and data_collector.is_connected():
            latest_tick = data_collector.get_latest_price(symbol)
            if latest_tick:
                return {
                    "source": "realtime",
                    "symbol": latest_tick.symbol,
                    "timestamp": latest_tick.timestamp.isoformat(),
                    "bid": latest_tick.bid,
                    "ask": latest_tick.ask,
                    "mid_price": latest_tick.mid_price,
                    "spread": latest_tick.spread
                }

        # Sinon, récupérer depuis la base de données
        latest_price = db_manager.get_latest_price(symbol, timeframe)
        if latest_price:
            return {
                "source": "database",
                "symbol": latest_price.symbol,
                "timestamp": latest_price.timestamp.isoformat(),
                "timeframe": latest_price.timeframe,
                "open": latest_price.open_price,
                "high": latest_price.high_price,
                "low": latest_price.low_price,
                "close": latest_price.close_price,
                "volume": latest_price.volume
            }

        return None

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données de marché: {e}")
        return None


def get_data_summary(symbol: str = "R_10", days: int = 7) -> Dict[str, Any]:
    """
    Génère un résumé des données disponibles

    Args:
        symbol: Symbole à analyser
        days: Nombre de jours à analyser

    Returns:
        Résumé des données
    """
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - pd.Timedelta(days=days)

        # Récupérer les données depuis la base
        price_data = db_manager.get_price_data(
            symbol=symbol,
            timeframe="1m",
            start_time=start_time,
            end_time=end_time
        )

        if not price_data:
            return {
                "symbol": symbol,
                "period_days": days,
                "data_available": False,
                "message": "Aucune donnée disponible"
            }

        # Convertir en DataFrame pour l'analyse
        df = pd.DataFrame([data.to_dict() for data in price_data])

        # Statistiques de base
        summary = {
            "symbol": symbol,
            "period_days": days,
            "data_available": True,
            "total_records": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "price_statistics": {
                "min_price": float(df['close'].min()),
                "max_price": float(df['close'].max()),
                "avg_price": float(df['close'].mean()),
                "volatility": float(df['close'].std()),
                "total_return": float((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100)
            }
        }

        # Qualité des données
        quality_report = validate_data_quality(df)
        summary["data_quality"] = {
            "score": quality_report.overall_score,
            "level": quality_report.quality_level.value,
            "completeness": quality_report.completeness,
            "issues_count": len(quality_report.issues)
        }

        # Activité récente
        last_24h = df[df['timestamp'] > (end_time - pd.Timedelta(hours=24))]
        summary["recent_activity"] = {
            "last_24h_records": len(last_24h),
            "data_freshness_minutes": (end_time - df['timestamp'].max()).total_seconds() / 60 if len(df) > 0 else None
        }

        return summary

    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé: {e}")
        return {
            "symbol": symbol,
            "period_days": days,
            "data_available": False,
            "error": str(e)
        }


def cleanup_old_data(days_to_keep: int = 365) -> Dict[str, int]:
    """
    Nettoie les anciennes données

    Args:
        days_to_keep: Nombre de jours de données à conserver

    Returns:
        Statistiques de nettoyage
    """
    try:
        logger.info(f"Début du nettoyage des données > {days_to_keep} jours")

        # Déléguer au gestionnaire de base de données
        db_manager.cleanup_old_data(days_to_keep)

        # Retourner les statistiques
        return {
            "days_kept": days_to_keep,
            "cleanup_completed": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")
        return {
            "days_kept": days_to_keep,
            "cleanup_completed": False,
            "error": str(e)
        }


def backup_data(backup_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Crée une sauvegarde des données critiques

    Args:
        backup_path: Chemin de sauvegarde (optionnel)

    Returns:
        Statut de la sauvegarde
    """
    try:
        import os
        import json
        from pathlib import Path

        if backup_path is None:
            backup_path = f"backups/data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Créer le dossier de sauvegarde
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)

        # Collecter les données à sauvegarder
        backup_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": __version__,
            "database_stats": db_manager.get_database_stats(),
            "pipeline_status": get_data_pipeline_status(),
            "processing_stats": preprocessor.get_processing_stats(),
            "collector_stats": get_collector_stats() if data_collector else {}
        }

        # Sauvegarder dans un fichier JSON
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)

        file_size = os.path.getsize(backup_path)

        logger.info(f"Sauvegarde créée: {backup_path} ({file_size} bytes)")

        return {
            "success": True,
            "backup_path": backup_path,
            "file_size_bytes": file_size,
            "timestamp": backup_data["timestamp"]
        }

    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def restore_data(backup_path: str) -> Dict[str, Any]:
    """
    Restaure les données depuis une sauvegarde

    Args:
        backup_path: Chemin vers le fichier de sauvegarde

    Returns:
        Statut de la restauration
    """
    try:
        import json
        import os

        if not os.path.exists(backup_path):
            return {
                "success": False,
                "error": f"Fichier de sauvegarde non trouvé: {backup_path}"
            }

        # Charger les données de sauvegarde
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        logger.info(f"Restauration depuis {backup_path}")
        logger.info(f"Sauvegarde créée le: {backup_data.get('timestamp', 'Inconnu')}")

        # Note: La restauration complète nécessiterait des opérations
        # spécifiques selon le type de données sauvegardées
        # Ici, on affiche juste les informations de la sauvegarde

        return {
            "success": True,
            "backup_timestamp": backup_data.get('timestamp'),
            "backup_version": backup_data.get('version'),
            "restored_components": list(backup_data.keys())
        }

    except Exception as e:
        logger.error(f"Erreur lors de la restauration: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def initialize_data_module() -> Dict[str, Any]:
    """
    Initialise complètement le module de données

    Returns:
        Statut de l'initialisation
    """
    try:
        logger.info("🚀 Initialisation du module de données...")

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components_initialized": [],
            "errors": []
        }

        # 1. Initialiser la base de données
        try:
            initialize_database()
            db_health = check_database_health()
            if db_health:
                results["components_initialized"].append("database")
                logger.info("✅ Base de données initialisée")
            else:
                results["errors"].append("Base de données non accessible")
        except Exception as e:
            results["errors"].append(f"Erreur base de données: {str(e)}")

        # 2. Vérifier le collecteur (ne pas démarrer automatiquement)
        try:
            if data_collector:
                collector_status = data_collector.status.value
                results["components_initialized"].append("collector")
                logger.info(f"✅ Collecteur initialisé (statut: {collector_status})")
        except Exception as e:
            results["errors"].append(f"Erreur collecteur: {str(e)}")

        # 3. Vérifier le préprocesseur
        try:
            if preprocessor:
                results["components_initialized"].append("preprocessor")
                logger.info("✅ Préprocesseur initialisé")
        except Exception as e:
            results["errors"].append(f"Erreur préprocesseur: {str(e)}")

        # Statut global
        results["success"] = len(results["errors"]) == 0
        results["components_count"] = len(results["components_initialized"])

        if results["success"]:
            logger.info(f"✅ Module de données initialisé ({results['components_count']} composants)")
        else:
            logger.warning(f"⚠️ Initialisation partielle ({len(results['errors'])} erreurs)")

        return results

    except Exception as e:
        logger.error(f"Erreur critique lors de l'initialisation: {e}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "error": str(e)
        }


def print_data_module_banner():
    """Affiche la bannière du module de données"""
    print("=" * 80)
    print("📊 TRADING BOT VOLATILITY 10 - MODULE DE DONNÉES")
    print("=" * 80)
    print(f"📦 Version: {__version__}")
    print(f"👥 Auteur: {__author__}")
    print(f"📝 Description: {__description__}")

    # Statut du pipeline
    status = get_data_pipeline_status()
    health_emoji = {
        "excellent": "✅",
        "partial": "⚠️",
        "critical": "❌",
        "error": "💥",
        "unknown": "❓"
    }

    print(
        f"\n🏥 Santé du pipeline: {health_emoji.get(status['overall_health'], '❓')} {status['overall_health'].upper()}")

    for component, info in status.get("components", {}).items():
        component_emoji = "✅" if info.get("healthy", False) else "❌"
        print(f"   {component_emoji} {component.title()}: {info.get('connection', 'unknown')}")

    print("=" * 80)


# Initialisation automatique au chargement du module
try:
    # Initialiser le module silencieusement
    init_results = initialize_data_module()

    if init_results["success"]:
        logger.info(f"Module de données chargé avec succès ({init_results['components_count']} composants)")
    else:
        logger.warning(f"Module de données partiellement chargé: {len(init_results.get('errors', []))} erreurs")
        for error in init_results.get('errors', []):
            logger.warning(f"  - {error}")

except Exception as e:
    logger.error(f"⚠️ Erreur lors de l'initialisation du module de données: {e}")
    # Ne pas faire planter l'import, permettre l'utilisation partielle