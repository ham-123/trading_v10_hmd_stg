"""
Module de configuration pour le Trading Bot Volatility 10
Centralise toute la configuration et la gestion des secrets

Usage:
    from config import config, get_secret_manager, get_api_manager
    from config import TradingMode, SYMBOL, TIMEFRAME
"""

# Imports des classes principales
from .settings import (
    TradingConfig,
    TradingMode,
    LogLevel,
    DatabaseConfig,
    RedisConfig,
    DerivAPIConfig,
    TradingParameters,
    TechnicalIndicators,
    AIModelConfig,
    MonitoringConfig,
    config,  # Instance globale
    setup_logging,
    # Constantes importantes
    SYMBOL,
    TIMEFRAME,
    TRADING_MODE,
    MAX_CAPITAL_PER_TRADE,
    STOP_LOSS_PIPS
)

from .secrets import (
    SecretManager,
    APIKeyManager,
    setup_secrets_manager,
    get_secret_manager,
    get_api_manager,
    create_env_template
)

# Version du module
__version__ = "1.0.0"

# Informations sur le module
__author__ = "Trading Bot Team"
__description__ = "Configuration et gestion des secrets pour le Trading Bot Volatility 10"

# Exports principaux
__all__ = [
    # Classes de configuration
    "TradingConfig",
    "TradingMode",
    "LogLevel",
    "DatabaseConfig",
    "RedisConfig",
    "DerivAPIConfig",
    "TradingParameters",
    "TechnicalIndicators",
    "AIModelConfig",
    "MonitoringConfig",

    # Classes de sécurité
    "SecretManager",
    "APIKeyManager",

    # Instance globale et fonctions
    "config",
    "setup_logging",
    "setup_secrets_manager",
    "get_secret_manager",
    "get_api_manager",
    "create_env_template",

    # Constantes importantes
    "SYMBOL",
    "TIMEFRAME",
    "TRADING_MODE",
    "MAX_CAPITAL_PER_TRADE",
    "STOP_LOSS_PIPS",

    # Métadonnées
    "__version__",
    "__author__",
    "__description__"
]


def validate_environment() -> tuple[bool, list[str]]:
    """
    Valide l'environnement complet (configuration + secrets)

    Returns:
        Tuple (is_valid, errors_list)
    """
    errors = []

    try:
        # Valider la configuration principale
        config._validate_config()
    except ValueError as e:
        errors.append(f"Configuration invalide: {str(e)}")

    try:
        # Valider les gestionnaires de secrets
        secret_manager = get_secret_manager()
        api_manager = get_api_manager()

        # Vérifier les clés API critiques
        if config.trading_mode == TradingMode.LIVE:
            deriv_creds = api_manager.get_deriv_credentials()
            if not deriv_creds or not deriv_creds.get("api_token"):
                errors.append("Token API Deriv requis pour le trading en live")

            # Valider le token Deriv si possible
            if not api_manager.validate_deriv_token():
                errors.append("Token API Deriv invalide")

        # Vérifier les alertes si activées
        if config.monitoring.telegram_alerts_enabled:
            telegram_creds = api_manager.get_telegram_credentials()
            if not telegram_creds or not telegram_creds.get("bot_token"):
                errors.append("Token Telegram requis pour les alertes")

    except Exception as e:
        errors.append(f"Erreur dans la gestion des secrets: {str(e)}")

    return len(errors) == 0, errors


def print_startup_banner():
    """Affiche la bannière de démarrage avec les informations de configuration"""
    print("=" * 80)
    print("🤖 TRADING BOT VOLATILITY 10 - SYSTÈME DE CONFIGURATION")
    print("=" * 80)
    print(f"📊 Version: {__version__}")
    print(f"⚙️ Mode: {config.trading_mode.value.upper()}")
    print(f"📈 Instrument: {config.trading.symbol}")
    print(f"⏱️ Timeframe: {config.trading.timeframe}")
    print(f"💰 Capital max/trade: {config.trading.max_capital_per_trade_pct * 100}%")
    print(f"🛡️ Stop Loss: {config.trading.stop_loss_pips} pips")
    print(f"🎯 Take Profit: 1:{config.trading.take_profit_ratio}")
    print(f"🧠 IA Confidence: {config.ai_model.min_confidence_threshold * 100}%")
    print(f"📡 Base de données: {'✅' if config.database.url else '❌'}")

    # Vérifier l'environnement
    is_valid, errors = validate_environment()

    if is_valid:
        print("✅ ENVIRONNEMENT VALIDÉ")
    else:
        print("❌ ERREURS DÉTECTÉES:")
        for error in errors:
            print(f"   • {error}")

    print("=" * 80)


def get_configuration_summary() -> dict:
    """
    Retourne un résumé complet de la configuration pour debugging

    Returns:
        Dictionnaire avec toutes les informations de configuration
    """
    secret_manager = get_secret_manager()
    api_manager = get_api_manager()

    return {
        "module_info": {
            "version": __version__,
            "author": __author__,
            "description": __description__
        },
        "configuration": config.get_config_summary(),
        "secrets": {
            "secrets_count": len(secret_manager.list_secrets()),
            "api_keys_loaded": len(api_manager.api_keys),
            "deriv_configured": bool(api_manager.get_deriv_credentials()),
            "telegram_configured": bool(api_manager.get_telegram_credentials())
        },
        "environment": {
            "is_valid": validate_environment()[0],
            "errors_count": len(validate_environment()[1])
        }
    }


# Initialisation automatique au chargement du module
try:
    # Configurer le logging
    logger = setup_logging()
    logger.info("Module de configuration chargé avec succès")

    # Valider l'environnement en mode silencieux
    is_valid, errors = validate_environment()
    if not is_valid:
        logger.warning(f"Environnement partiellement configuré: {len(errors)} erreurs")
        for error in errors:
            logger.warning(f"Configuration: {error}")
    else:
        logger.info("Environnement entièrement validé")

except Exception as e:
    print(f"⚠️ Erreur lors de l'initialisation du module config: {e}")
    # Ne pas faire planter l'import, permettre l'utilisation partielle