"""
Configuration principale du système de trading automatisé
Volatility 10 Trading Bot - settings.py
"""

import os
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class TradingMode(Enum):
    """Modes de trading disponibles"""
    PAPER = "paper"  # Simulation
    LIVE = "live"  # Trading réel
    BACKTEST = "backtest"  # Test historique


class LogLevel(Enum):
    """Niveaux de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Configuration base de données"""
    url: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/trading_bot")
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False  # True pour debug SQL


@dataclass
class RedisConfig:
    """Configuration Redis"""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True


@dataclass
class DerivAPIConfig:
    """Configuration API Deriv"""
    ws_url: str = "wss://ws.binaryws.com/websockets/v3"
    app_id: str = os.getenv("DERIV_APP_ID", "1089")
    api_token: str = os.getenv("DERIV_API_TOKEN", "")
    connection_timeout: int = 30
    reconnect_attempts: int = 5
    reconnect_delay: int = 5
    heartbeat_interval: int = 30


@dataclass
class TradingParameters:
    """Paramètres de trading"""
    # Instrument
    symbol: str = "R_10"  # Volatility 10 Index
    timeframe: str = "1m"  # 1 minute

    # Gestion du capital
    max_capital_per_trade_pct: float = 0.02  # 2% du capital total
    min_trade_amount: float = 0.35  # Minimum Deriv
    max_trade_amount: float = 1000.0
    initial_capital: float = 1000.0

    # Gestion des risques
    stop_loss_pips: int = 15
    take_profit_ratio: float = 2.0  # Risk/Reward 1:2
    max_trades_per_day: int = 20
    max_consecutive_losses: int = 3
    max_drawdown_pct: float = 0.10  # 10%
    emergency_stop_loss_pct: float = 0.05  # 5%

    # Contraintes temporelles
    trading_start_hour: int = 0  # 00:00 UTC
    trading_end_hour: int = 24  # 24:00 UTC
    avoid_news_events: bool = True
    weekend_trading: bool = True  # Synthetics tradent 24/7


@dataclass
class TechnicalIndicators:
    """Configuration des indicateurs techniques"""
    # Moyennes mobiles
    sma_periods: List[int] = None
    ema_periods: List[int] = None

    # Oscillateurs
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Volume
    volume_sma_period: int = 20

    # Support/Résistance
    sr_lookback: int = 50
    sr_strength: int = 3

    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26]


@dataclass
class AIModelConfig:
    """Configuration du modèle IA"""
    # Architecture LSTM
    lstm_units: List[int] = None
    dropout_rate: float = 0.2
    lookback_periods: int = 100
    prediction_horizon: int = 5  # minutes

    # Entraînement
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    learning_rate: float = 0.001

    # Features
    use_technical_indicators: bool = True
    use_price_action: bool = True
    use_volume: bool = True
    normalize_features: bool = True

    # Prédictions
    min_confidence_threshold: float = 0.65
    model_retrain_frequency_hours: int = 24
    model_validation_frequency_hours: int = 6

    # Performance
    min_accuracy_threshold: float = 0.55
    min_sharpe_ratio: float = 1.0

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [64, 32, 16]


@dataclass
class MonitoringConfig:
    """Configuration monitoring et alertes"""
    # Logs
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file_path: str = "logs/trading_bot.log"
    log_max_size_mb: int = 10
    log_backup_count: int = 5

    # Dashboard
    dashboard_port: int = 8050
    dashboard_host: str = "0.0.0.0"
    update_interval_seconds: int = 5

    # Alertes Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_alerts_enabled: bool = True

    # Alertes Email
    email_smtp_server: str = os.getenv("EMAIL_SMTP_SERVER", "")
    email_smtp_port: int = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    email_username: str = os.getenv("EMAIL_USERNAME", "")
    email_password: str = os.getenv("EMAIL_PASSWORD", "")
    email_alerts_enabled: bool = False

    # Métriques
    save_metrics_to_db: bool = True
    metrics_retention_days: int = 365


class TradingConfig:
    """Configuration principale du système de trading"""

    def __init__(self):
        # Mode de trading
        self.trading_mode = TradingMode(os.getenv("TRADING_MODE", "paper"))

        # Configurations des modules
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.deriv_api = DerivAPIConfig()
        self.trading = TradingParameters()
        self.indicators = TechnicalIndicators()
        self.ai_model = AIModelConfig()
        self.monitoring = MonitoringConfig()

        # Validation automatique
        self._validate_config()

    def _validate_config(self) -> None:
        """Valide la configuration au démarrage"""
        errors = []
        warnings = []

        # Validation API Deriv
        if not self.deriv_api.api_token:
            if self.trading_mode == TradingMode.LIVE:
                errors.append("❌ DERIV_API_TOKEN requis pour le trading en live")
            else:
                warnings.append("⚠️ DERIV_API_TOKEN manquant (OK pour simulation)")

        # Validation base de données
        if not self.database.url:
            errors.append("❌ DATABASE_URL requis")

        # Validation paramètres de trading
        if self.trading.max_capital_per_trade_pct <= 0 or self.trading.max_capital_per_trade_pct > 0.1:
            warnings.append("⚠️ Capital par trade recommandé entre 1-10%")

        if self.trading.stop_loss_pips <= 0:
            errors.append("❌ Stop loss doit être positif")

        if self.trading.take_profit_ratio < 1.0:
            warnings.append("⚠️ Ratio risk/reward < 1:1 non recommandé")

        # Validation IA
        if self.ai_model.min_confidence_threshold < 0.5 or self.ai_model.min_confidence_threshold > 0.95:
            warnings.append("⚠️ Seuil de confiance IA recommandé entre 50-95%")

        # Validation monitoring
        if self.trading_mode == TradingMode.LIVE and not self.monitoring.telegram_bot_token:
            warnings.append("⚠️ Telegram recommandé pour alertes en live")

        # Affichage des résultats
        if errors:
            print("\n🔴 ERREURS DE CONFIGURATION:")
            for error in errors:
                print(f"  {error}")
            raise ValueError("Configuration invalide - voir erreurs ci-dessus")

        if warnings:
            print("\n🟡 AVERTISSEMENTS DE CONFIGURATION:")
            for warning in warnings:
                print(f"  {warning}")

        print("✅ Configuration validée avec succès")
        print(f"📊 Mode: {self.trading_mode.value}")
        print(f"📈 Instrument: {self.trading.symbol}")
        print(f"⏱️ Timeframe: {self.trading.timeframe}")
        print(f"💰 Capital max par trade: {self.trading.max_capital_per_trade_pct * 100}%")

    def get_config_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la configuration"""
        return {
            "trading_mode": self.trading_mode.value,
            "symbol": self.trading.symbol,
            "timeframe": self.trading.timeframe,
            "max_capital_per_trade": f"{self.trading.max_capital_per_trade_pct * 100}%",
            "stop_loss_pips": self.trading.stop_loss_pips,
            "take_profit_ratio": f"1:{self.trading.take_profit_ratio}",
            "max_trades_per_day": self.trading.max_trades_per_day,
            "ai_confidence_threshold": f"{self.ai_model.min_confidence_threshold * 100}%",
            "indicators_count": len(self.indicators.sma_periods) + len(self.indicators.ema_periods) + 4,
            "database_configured": bool(self.database.url),
            "telegram_alerts": bool(self.monitoring.telegram_bot_token),
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Exporte toute la configuration en dictionnaire"""
        return {
            "trading_mode": self.trading_mode.value,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "deriv_api": {k: v for k, v in self.deriv_api.__dict__.items() if k != "api_token"},
            "trading": self.trading.__dict__,
            "indicators": self.indicators.__dict__,
            "ai_model": self.ai_model.__dict__,
            "monitoring": {k: v for k, v in self.monitoring.__dict__.items()
                           if not any(secret in k.lower() for secret in ["token", "password"])}
        }


# Instance globale de configuration
config = TradingConfig()

# Export des constantes importantes
SYMBOL = config.trading.symbol
TIMEFRAME = config.trading.timeframe
TRADING_MODE = config.trading_mode
MAX_CAPITAL_PER_TRADE = config.trading.max_capital_per_trade_pct
STOP_LOSS_PIPS = config.trading.stop_loss_pips


# Configuration logging
def setup_logging():
    """Configure le système de logging"""
    log_level = getattr(logging, config.monitoring.log_level.upper())

    # Créer le dossier de logs si inexistant
    os.makedirs(os.path.dirname(config.monitoring.log_file_path), exist_ok=True)

    # Configuration du logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.monitoring.log_file_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("TradingBot")


if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Test de la configuration...")
    print("\n📋 Résumé de la configuration:")

    summary = config.get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n📊 Configuration complète exportée vers dict: {len(config.export_to_dict())} sections")
    print("✅ Test terminé avec succès !")