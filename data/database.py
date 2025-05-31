"""
Gestion de la base de données pour le Trading Bot Volatility 10
Utilise SQLAlchemy pour la gestion des données OHLCV, trades, signaux, etc.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Boolean,
    Text, ForeignKey, Index, UniqueConstraint, CheckConstraint,
    func, and_, or_, desc, asc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import text
import uuid
import json

from config import config

# Configuration du logger
logger = logging.getLogger(__name__)

# Base pour tous les modèles
Base = declarative_base()


# =============================================================================
# MODÈLES DE DONNÉES
# =============================================================================

class PriceData(Base):
    """Données OHLCV pour Volatility 10"""
    __tablename__ = 'price_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)

    # OHLCV data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)

    # Métadonnées
    spread = Column(Float)
    tick_count = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    # Contraintes
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', 'timeframe', name='uq_price_data_symbol_time'),
        CheckConstraint('high_price >= low_price', name='check_high_low'),
        CheckConstraint('high_price >= open_price', name='check_high_open'),
        CheckConstraint('high_price >= close_price', name='check_high_close'),
        CheckConstraint('low_price <= open_price', name='check_low_open'),
        CheckConstraint('low_price <= close_price', name='check_low_close'),
        Index('idx_price_data_symbol_time', 'symbol', 'timestamp'),
        Index('idx_price_data_timeframe', 'timeframe'),
    )

    def __repr__(self):
        return f"<PriceData({self.symbol}, {self.timestamp}, O:{self.open_price}, C:{self.close_price})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume,
            'spread': self.spread,
            'tick_count': self.tick_count
        }


class TechnicalIndicators(Base):
    """Indicateurs techniques calculés"""
    __tablename__ = 'technical_indicators'

    id = Column(Integer, primary_key=True, autoincrement=True)
    price_data_id = Column(Integer, ForeignKey('price_data.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Moyennes mobiles
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)

    # Oscillateurs
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)

    # Bollinger Bands
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    bb_position = Column(Float)  # Position du prix dans les bandes

    # Volume
    volume_sma = Column(Float)
    volume_ratio = Column(Float)

    # Support/Résistance
    support_level = Column(Float)
    resistance_level = Column(Float)
    sr_strength = Column(Float)

    # Patterns et signaux
    trend_direction = Column(String(10))  # 'up', 'down', 'sideways'
    trend_strength = Column(Float)
    volatility = Column(Float)

    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    # Relation
    price_data = relationship("PriceData", backref="indicators")

    __table_args__ = (
        Index('idx_indicators_timestamp', 'timestamp'),
    )


class TradingSignals(Base):
    """Signaux de trading générés"""
    __tablename__ = 'trading_signals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(50), unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)

    # Signal
    signal_type = Column(String(10), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    confidence = Column(Float, nullable=False)
    strength = Column(Float)

    # Sources du signal
    technical_score = Column(Float)  # Score des indicateurs techniques
    ai_prediction = Column(Float)  # Prédiction IA
    pattern_score = Column(Float)  # Score des patterns

    # Paramètres de trading
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    position_size = Column(Float)
    risk_reward_ratio = Column(Float)

    # Métadonnées
    indicators_used = Column(JSONB)  # Liste des indicateurs utilisés
    reasoning = Column(Text)  # Explication du signal
    market_conditions = Column(JSONB)  # Conditions du marché

    # Statut
    status = Column(String(20), default='active')  # 'active', 'executed', 'expired', 'cancelled'
    expires_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    __table_args__ = (
        CheckConstraint("signal_type IN ('BUY', 'SELL', 'HOLD')", name='check_signal_type'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence'),
        Index('idx_signals_timestamp', 'timestamp'),
        Index('idx_signals_symbol_type', 'symbol', 'signal_type'),
    )


class Trades(Base):
    """Historique des trades exécutés"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    signal_id = Column(String(50), ForeignKey('trading_signals.signal_id'))

    # Informations de base
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'BUY', 'SELL'
    status = Column(String(20), nullable=False)  # 'open', 'closed', 'cancelled'

    # Prix et volumes
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    position_size = Column(Float, nullable=False)

    # Gestion des risques
    stop_loss = Column(Float)
    take_profit = Column(Float)
    actual_stop_loss = Column(Float)
    actual_take_profit = Column(Float)

    # Résultats
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    net_profit_loss = Column(Float)

    # Timing
    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True))
    duration_minutes = Column(Integer)

    # Métadonnées
    market_conditions = Column(JSONB)
    exit_reason = Column(String(50))  # 'take_profit', 'stop_loss', 'manual', 'timeout'
    notes = Column(Text)

    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=datetime.now(timezone.utc))

    # Relation
    signal = relationship("TradingSignals", backref="trades")

    __table_args__ = (
        CheckConstraint("trade_type IN ('BUY', 'SELL')", name='check_trade_type'),
        CheckConstraint("status IN ('open', 'closed', 'cancelled')", name='check_trade_status'),
        Index('idx_trades_symbol_status', 'symbol', 'status'),
        Index('idx_trades_entry_time', 'entry_time'),
    )


class PerformanceMetrics(Base):
    """Métriques de performance du système"""
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    period = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly'

    # Métriques de trading
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)

    # Profits et pertes
    total_profit_loss = Column(Float, default=0.0)
    gross_profit = Column(Float, default=0.0)
    gross_loss = Column(Float, default=0.0)
    net_profit_loss = Column(Float, default=0.0)

    # Ratios de performance
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_pct = Column(Float)
    recovery_factor = Column(Float)

    # Métriques IA
    ai_prediction_accuracy = Column(Float)
    signal_accuracy = Column(Float)
    false_positive_rate = Column(Float)

    # Capital et position
    account_balance = Column(Float)
    equity = Column(Float)
    margin_used = Column(Float)
    free_margin = Column(Float)

    # Métadonnées
    data_quality_score = Column(Float)
    system_uptime_pct = Column(Float)

    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    __table_args__ = (
        CheckConstraint("period IN ('daily', 'weekly', 'monthly')", name='check_period'),
        UniqueConstraint('timestamp', 'period', name='uq_metrics_timestamp_period'),
    )


class SystemLogs(Base):
    """Logs du système pour debugging et monitoring"""
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    level = Column(String(10), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)

    # Contexte supplémentaire
    function_name = Column(String(100))
    line_number = Column(Integer)
    trade_id = Column(String(50))
    signal_id = Column(String(50))

    # Données structurées
    extra_data = Column(JSONB)
    stack_trace = Column(Text)

    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    __table_args__ = (
        CheckConstraint("level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name='check_log_level'),
        Index('idx_logs_level_timestamp', 'level', 'timestamp'),
        Index('idx_logs_module', 'module'),
    )


# =============================================================================
# GESTIONNAIRE DE BASE DE DONNÉES
# =============================================================================

class DatabaseManager:
    """Gestionnaire principal de la base de données"""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or config.database.url
        self.engine = None
        self.SessionLocal = None
        self.Session = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialise la connexion à la base de données"""
        try:
            # Configuration du moteur
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=config.database.pool_size,
                max_overflow=config.database.max_overflow,
                pool_timeout=config.database.pool_timeout,
                pool_recycle=config.database.pool_recycle,
                echo=config.database.echo
            )

            # Configuration des sessions
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )

            # Session thread-safe
            self.Session = scoped_session(self.SessionLocal)

            logger.info("Connexion à la base de données établie")

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise

    def create_tables(self):
        """Crée toutes les tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Tables créées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création des tables: {e}")
            raise

    def drop_tables(self):
        """Supprime toutes les tables (ATTENTION: destructif)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("Toutes les tables ont été supprimées")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des tables: {e}")
            raise

    @contextmanager
    def get_session(self):
        """Context manager pour les sessions de base de données"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Vérifie la santé de la connexion à la base de données"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # =============================================================================
    # MÉTHODES CRUD POUR PRICE DATA
    # =============================================================================

    def save_price_data(self, symbol: str, timestamp: datetime, timeframe: str,
                        open_price: float, high_price: float, low_price: float,
                        close_price: float, volume: float = 0.0,
                        spread: Optional[float] = None) -> Optional[PriceData]:
        """Sauvegarde des données de prix"""
        try:
            with self.get_session() as session:
                # Vérifier si les données existent déjà
                existing = session.query(PriceData).filter(
                    and_(
                        PriceData.symbol == symbol,
                        PriceData.timestamp == timestamp,
                        PriceData.timeframe == timeframe
                    )
                ).first()

                if existing:
                    # Mettre à jour les données existantes
                    existing.open_price = open_price
                    existing.high_price = high_price
                    existing.low_price = low_price
                    existing.close_price = close_price
                    existing.volume = volume
                    existing.spread = spread
                    return existing
                else:
                    # Créer une nouvelle entrée
                    price_data = PriceData(
                        symbol=symbol,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open_price=open_price,
                        high_price=high_price,
                        low_price=low_price,
                        close_price=close_price,
                        volume=volume,
                        spread=spread
                    )
                    session.add(price_data)
                    session.flush()  # Pour obtenir l'ID
                    return price_data

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données de prix: {e}")
            return None

    def get_price_data(self, symbol: str, timeframe: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: Optional[int] = None) -> List[PriceData]:
        """Récupère les données de prix"""
        try:
            with self.get_session() as session:
                query = session.query(PriceData).filter(
                    and_(
                        PriceData.symbol == symbol,
                        PriceData.timeframe == timeframe
                    )
                )

                if start_time:
                    query = query.filter(PriceData.timestamp >= start_time)
                if end_time:
                    query = query.filter(PriceData.timestamp <= end_time)

                query = query.order_by(PriceData.timestamp)

                if limit:
                    query = query.limit(limit)

                return query.all()

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de prix: {e}")
            return []

    def get_latest_price(self, symbol: str, timeframe: str) -> Optional[PriceData]:
        """Récupère le dernier prix disponible"""
        try:
            with self.get_session() as session:
                return session.query(PriceData).filter(
                    and_(
                        PriceData.symbol == symbol,
                        PriceData.timeframe == timeframe
                    )
                ).order_by(desc(PriceData.timestamp)).first()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du dernier prix: {e}")
            return None

    # =============================================================================
    # MÉTHODES CRUD POUR TRADING SIGNALS
    # =============================================================================

    def save_trading_signal(self, signal_data: Dict[str, Any]) -> Optional[TradingSignals]:
        """Sauvegarde un signal de trading"""
        try:
            with self.get_session() as session:
                signal = TradingSignals(
                    signal_id=signal_data.get('signal_id', str(uuid.uuid4())),
                    timestamp=signal_data['timestamp'],
                    symbol=signal_data['symbol'],
                    signal_type=signal_data['signal_type'],
                    confidence=signal_data['confidence'],
                    strength=signal_data.get('strength'),
                    technical_score=signal_data.get('technical_score'),
                    ai_prediction=signal_data.get('ai_prediction'),
                    pattern_score=signal_data.get('pattern_score'),
                    entry_price=signal_data.get('entry_price'),
                    stop_loss=signal_data.get('stop_loss'),
                    take_profit=signal_data.get('take_profit'),
                    position_size=signal_data.get('position_size'),
                    risk_reward_ratio=signal_data.get('risk_reward_ratio'),
                    indicators_used=signal_data.get('indicators_used'),
                    reasoning=signal_data.get('reasoning'),
                    market_conditions=signal_data.get('market_conditions'),
                    expires_at=signal_data.get('expires_at')
                )
                session.add(signal)
                session.flush()
                return signal
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du signal: {e}")
            return None

    def get_active_signals(self, symbol: str) -> List[TradingSignals]:
        """Récupère les signaux actifs pour un symbole"""
        try:
            with self.get_session() as session:
                return session.query(TradingSignals).filter(
                    and_(
                        TradingSignals.symbol == symbol,
                        TradingSignals.status == 'active',
                        or_(
                            TradingSignals.expires_at.is_(None),
                            TradingSignals.expires_at > datetime.now(timezone.utc)
                        )
                    )
                ).order_by(desc(TradingSignals.timestamp)).all()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des signaux actifs: {e}")
            return []

    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur la base de données"""
        try:
            with self.get_session() as session:
                stats = {}

                # Compter les enregistrements dans chaque table
                stats['price_data_count'] = session.query(PriceData).count()
                stats['signals_count'] = session.query(TradingSignals).count()
                stats['trades_count'] = session.query(Trades).count()
                stats['indicators_count'] = session.query(TechnicalIndicators).count()

                # Dernière mise à jour
                latest_price = session.query(PriceData).order_by(desc(PriceData.timestamp)).first()
                stats['latest_data_timestamp'] = latest_price.timestamp if latest_price else None

                # Taille de la base de données (PostgreSQL)
                if 'postgresql' in self.database_url:
                    db_name = self.database_url.split('/')[-1]
                    result = session.execute(text(f"""
                        SELECT pg_size_pretty(pg_database_size('{db_name}')) as size
                    """)).first()
                    stats['database_size'] = result[0] if result else 'Unknown'

                return stats

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques: {e}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 365):
        """Nettoie les anciennes données"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            with self.get_session() as session:
                # Supprimer les anciens logs
                deleted_logs = session.query(SystemLogs).filter(
                    SystemLogs.timestamp < cutoff_date
                ).delete()

                logger.info(f"Supprimé {deleted_logs} anciens logs")

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")


# Instance globale
db_manager = DatabaseManager()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_db_session():
    """Retourne une nouvelle session de base de données"""
    return db_manager.get_session()


def initialize_database():
    """Initialise la base de données (crée les tables)"""
    db_manager.create_tables()


def check_database_health() -> bool:
    """Vérifie la santé de la base de données"""
    return db_manager.health_check()


if __name__ == "__main__":
    # Test de la base de données
    print("🗄️ Test du gestionnaire de base de données...")

    try:
        # Créer les tables
        print("📊 Création des tables...")
        initialize_database()

        # Test de santé
        print("🏥 Vérification de la santé...")
        health = check_database_health()
        print(f"✅ Santé de la DB: {'OK' if health else 'ERREUR'}")

        # Statistiques
        print("📈 Statistiques de la base de données:")
        stats = db_manager.get_database_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("✅ Test de la base de données réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test de la base de données échoué: {e}")