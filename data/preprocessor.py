"""
Préprocesseur de données pour le Trading Bot Volatility 10
Nettoyage, validation et préparation des données pour l'IA
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import scipy.stats as stats

from config import config
from .database import db_manager, PriceData

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DataQuality(Enum):
    """Niveaux de qualité des données"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DataQualityReport:
    """Rapport de qualité des données"""
    overall_score: float
    completeness: float
    consistency: float
    validity: float
    timeliness: float
    outlier_percentage: float
    missing_data_percentage: float
    duplicate_percentage: float
    issues: List[str]
    recommendations: List[str]

    @property
    def quality_level(self) -> DataQuality:
        """Détermine le niveau de qualité"""
        if self.overall_score >= 0.9:
            return DataQuality.EXCELLENT
        elif self.overall_score >= 0.8:
            return DataQuality.GOOD
        elif self.overall_score >= 0.6:
            return DataQuality.FAIR
        elif self.overall_score >= 0.4:
            return DataQuality.POOR
        else:
            return DataQuality.CRITICAL


@dataclass
class FeatureSet:
    """Ensemble de features pour l'IA"""
    price_features: pd.DataFrame
    technical_features: pd.DataFrame
    volume_features: pd.DataFrame
    time_features: pd.DataFrame
    market_features: pd.DataFrame
    target_values: pd.Series
    feature_names: List[str]
    timestamp_index: pd.DatetimeIndex


class DataPreprocessor:
    """Préprocesseur principal des données"""

    def __init__(self):
        self.scaler_price = StandardScaler()
        self.scaler_volume = RobustScaler()
        self.scaler_technical = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)

        # Configuration
        self.lookback_periods = config.ai_model.lookback_periods
        self.prediction_horizon = config.ai_model.prediction_horizon

        # Cache des données nettoyées
        self.cleaned_data_cache = {}
        self.feature_cache = {}

        # Statistiques
        self.processing_stats = {
            'records_processed': 0,
            'outliers_removed': 0,
            'missing_values_filled': 0,
            'duplicates_removed': 0,
            'features_created': 0,
            'last_processing_time': None
        }

        logger.info("Préprocesseur de données initialisé")

    def validate_raw_data(self, data: pd.DataFrame) -> DataQualityReport:
        """Valide et évalue la qualité des données brutes"""
        try:
            issues = []
            recommendations = []

            # 1. Complétude des données
            total_records = len(data)
            missing_count = data.isnull().sum().sum()
            completeness = 1 - (missing_count / (total_records * len(data.columns)))

            if missing_count > 0:
                issues.append(f"{missing_count} valeurs manquantes détectées")
                recommendations.append("Appliquer l'imputation des valeurs manquantes")

            # 2. Cohérence des prix OHLC
            consistency_errors = 0
            if all(col in data.columns for col in ['open_price', 'high_price', 'low_price', 'close_price']):
                # High >= Low
                high_low_errors = (data['high_price'] < data['low_price']).sum()
                # High >= Open et Close
                high_open_errors = (data['high_price'] < data['open_price']).sum()
                high_close_errors = (data['high_price'] < data['close_price']).sum()
                # Low <= Open et Close
                low_open_errors = (data['low_price'] > data['open_price']).sum()
                low_close_errors = (data['low_price'] > data['close_price']).sum()

                consistency_errors = sum([high_low_errors, high_open_errors, high_close_errors,
                                          low_open_errors, low_close_errors])

                if consistency_errors > 0:
                    issues.append(f"{consistency_errors} erreurs de cohérence OHLC")
                    recommendations.append("Corriger ou supprimer les données incohérentes")

            consistency = 1 - (consistency_errors / max(1, total_records))

            # 3. Validité des données
            validity_errors = 0

            # Prix négatifs
            price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
            for col in price_columns:
                if col in data.columns:
                    negative_prices = (data[col] <= 0).sum()
                    validity_errors += negative_prices
                    if negative_prices > 0:
                        issues.append(f"{negative_prices} prix négatifs ou nuls dans {col}")

            # Volume négatif
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                validity_errors += negative_volume
                if negative_volume > 0:
                    issues.append(f"{negative_volume} volumes négatifs")

            validity = 1 - (validity_errors / max(1, total_records))

            # 4. Ponctualité (fraîcheur des données)
            timeliness = 1.0
            if 'timestamp' in data.columns:
                latest_timestamp = pd.to_datetime(data['timestamp']).max()
                current_time = datetime.now(timezone.utc)

                if pd.notna(latest_timestamp):
                    if latest_timestamp.tz is None:
                        latest_timestamp = latest_timestamp.tz_localize(timezone.utc)

                    time_diff = (current_time - latest_timestamp).total_seconds()
                    # Pénalité si les données sont anciennes (> 5 minutes)
                    timeliness = max(0, 1 - (time_diff / 300))

                    if time_diff > 300:
                        issues.append(f"Données anciennes: {time_diff / 60:.1f} minutes")
                        recommendations.append("Vérifier la collecte de données en temps réel")

            # 5. Détection des outliers
            outlier_count = 0
            outlier_percentage = 0

            if 'close_price' in data.columns and len(data) > 10:
                close_prices = data['close_price'].dropna()
                if len(close_prices) > 0:
                    Q1 = close_prices.quantile(0.25)
                    Q3 = close_prices.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = (close_prices < lower_bound) | (close_prices > upper_bound)
                    outlier_count = outliers.sum()
                    outlier_percentage = outlier_count / len(close_prices)

                    if outlier_percentage > 0.05:  # Plus de 5%
                        issues.append(f"{outlier_percentage * 100:.1f}% d'outliers détectés")
                        recommendations.append("Examiner et traiter les valeurs aberrantes")

            # 6. Détection des doublons
            duplicates = data.duplicated().sum()
            duplicate_percentage = duplicates / max(1, total_records)

            if duplicates > 0:
                issues.append(f"{duplicates} doublons détectés")
                recommendations.append("Supprimer les enregistrements en double")

            # Score global
            overall_score = (completeness + consistency + validity + timeliness) / 4

            # Pénalités pour les problèmes majeurs
            if outlier_percentage > 0.1:  # Plus de 10%
                overall_score *= 0.8
            if duplicate_percentage > 0.05:  # Plus de 5%
                overall_score *= 0.9

            return DataQualityReport(
                overall_score=overall_score,
                completeness=completeness,
                consistency=consistency,
                validity=validity,
                timeliness=timeliness,
                outlier_percentage=outlier_percentage,
                missing_data_percentage=missing_count / (total_records * len(data.columns)),
                duplicate_percentage=duplicate_percentage,
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Erreur lors de la validation des données: {e}")
            return DataQualityReport(
                overall_score=0.0,
                completeness=0.0,
                consistency=0.0,
                validity=0.0,
                timeliness=0.0,
                outlier_percentage=1.0,
                missing_data_percentage=1.0,
                duplicate_percentage=0.0,
                issues=[f"Erreur de validation: {str(e)}"],
                recommendations=["Vérifier le format et la structure des données"]
            )

    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données de prix OHLCV"""
        try:
            logger.info(f"Nettoyage de {len(data)} enregistrements de prix")

            # Copie pour éviter de modifier l'original
            cleaned_data = data.copy()
            initial_count = len(cleaned_data)

            # 1. Supprimer les doublons
            duplicates_before = cleaned_data.duplicated().sum()
            cleaned_data = cleaned_data.drop_duplicates()
            duplicates_removed = duplicates_before - cleaned_data.duplicated().sum()
            self.processing_stats['duplicates_removed'] += duplicates_removed

            # 2. Valider et corriger les colonnes OHLC
            price_columns = ['open_price', 'high_price', 'low_price', 'close_price']

            for col in price_columns:
                if col in cleaned_data.columns:
                    # Supprimer les prix négatifs ou nuls
                    invalid_prices = (cleaned_data[col] <= 0) | cleaned_data[col].isnull()
                    cleaned_data = cleaned_data[~invalid_prices]

            # 3. Vérifier la cohérence OHLC et corriger si possible
            if all(col in cleaned_data.columns for col in price_columns):
                # Corriger les incohérences mineures
                cleaned_data = self._fix_ohlc_inconsistencies(cleaned_data)

                # Supprimer les enregistrements avec des incohérences majeures
                valid_ohlc = (
                        (cleaned_data['high_price'] >= cleaned_data['low_price']) &
                        (cleaned_data['high_price'] >= cleaned_data['open_price']) &
                        (cleaned_data['high_price'] >= cleaned_data['close_price']) &
                        (cleaned_data['low_price'] <= cleaned_data['open_price']) &
                        (cleaned_data['low_price'] <= cleaned_data['close_price'])
                )

                invalid_count = (~valid_ohlc).sum()
                if invalid_count > 0:
                    logger.warning(f"Suppression de {invalid_count} enregistrements avec OHLC incohérent")

                cleaned_data = cleaned_data[valid_ohlc]

            # 4. Traiter les outliers
            if 'close_price' in cleaned_data.columns:
                cleaned_data = self._remove_price_outliers(cleaned_data)

            # 5. Interpoler les valeurs manquantes
            cleaned_data = self._interpolate_missing_values(cleaned_data)

            # 6. Valider les volumes
            if 'volume' in cleaned_data.columns:
                # Remplacer les volumes négatifs par 0
                cleaned_data.loc[cleaned_data['volume'] < 0, 'volume'] = 0

                # Interpoler les volumes manquants
                cleaned_data['volume'] = cleaned_data['volume'].fillna(method='ffill').fillna(0)

            # 7. Assurer l'ordre chronologique
            if 'timestamp' in cleaned_data.columns:
                cleaned_data = cleaned_data.sort_values('timestamp')
                cleaned_data = cleaned_data.reset_index(drop=True)

            records_processed = len(cleaned_data)
            records_removed = initial_count - records_processed

            self.processing_stats['records_processed'] += records_processed

            logger.info(f"Nettoyage terminé: {records_processed} enregistrements conservés, "
                        f"{records_removed} supprimés")

            return cleaned_data

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des données: {e}")
            return data

    def _fix_ohlc_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Corrige les incohérences mineures dans les données OHLC"""
        try:
            fixed_data = data.copy()

            # Si High < Low, les inverser
            swap_mask = fixed_data['high_price'] < fixed_data['low_price']
            if swap_mask.any():
                logger.warning(f"Inversion de {swap_mask.sum()} valeurs High/Low")
                fixed_data.loc[swap_mask, ['high_price', 'low_price']] = \
                    fixed_data.loc[swap_mask, ['low_price', 'high_price']].values

            # Ajuster High pour être >= Open et Close
            high_too_low = (
                    (fixed_data['high_price'] < fixed_data['open_price']) |
                    (fixed_data['high_price'] < fixed_data['close_price'])
            )
            if high_too_low.any():
                fixed_data.loc[high_too_low, 'high_price'] = fixed_data.loc[high_too_low,
                ['open_price', 'close_price', 'high_price']].max(axis=1)

            # Ajuster Low pour être <= Open et Close
            low_too_high = (
                    (fixed_data['low_price'] > fixed_data['open_price']) |
                    (fixed_data['low_price'] > fixed_data['close_price'])
            )
            if low_too_high.any():
                fixed_data.loc[low_too_high, 'low_price'] = fixed_data.loc[low_too_high,
                ['open_price', 'close_price', 'low_price']].min(axis=1)

            return fixed_data

        except Exception as e:
            logger.error(f"Erreur lors de la correction OHLC: {e}")
            return data

    def _remove_price_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Supprime les outliers de prix"""
        try:
            if 'close_price' not in data.columns or len(data) < 10:
                return data

            outliers_removed = 0

            if method == 'iqr':
                # Méthode IQR
                Q1 = data['close_price'].quantile(0.25)
                Q3 = data['close_price'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (
                        (data['close_price'] < lower_bound) |
                        (data['close_price'] > upper_bound)
                )

            elif method == 'zscore':
                # Méthode Z-score
                z_scores = np.abs(stats.zscore(data['close_price']))
                outlier_mask = z_scores > 3

            else:
                return data

            outliers_removed = outlier_mask.sum()

            if outliers_removed > 0:
                logger.info(f"Suppression de {outliers_removed} outliers de prix")
                self.processing_stats['outliers_removed'] += outliers_removed
                return data[~outlier_mask]

            return data

        except Exception as e:
            logger.error(f"Erreur lors de la suppression des outliers: {e}")
            return data

    def _interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Interpole les valeurs manquantes"""
        try:
            if data.isnull().sum().sum() == 0:
                return data

            interpolated_data = data.copy()
            price_columns = ['open_price', 'high_price', 'low_price', 'close_price']

            # Interpolation linéaire pour les prix
            for col in price_columns:
                if col in interpolated_data.columns:
                    missing_before = interpolated_data[col].isnull().sum()
                    interpolated_data[col] = interpolated_data[col].interpolate(method='linear')

                    # Fill forward/backward pour les valeurs aux extrémités
                    interpolated_data[col] = interpolated_data[col].fillna(method='ffill').fillna(method='bfill')

                    missing_after = interpolated_data[col].isnull().sum()
                    filled = missing_before - missing_after
                    self.processing_stats['missing_values_filled'] += filled

            # Volume: fill forward puis 0
            if 'volume' in interpolated_data.columns:
                interpolated_data['volume'] = interpolated_data['volume'].fillna(method='ffill').fillna(0)

            return interpolated_data

        except Exception as e:
            logger.error(f"Erreur lors de l'interpolation: {e}")
            return data

    def create_features(self, data: pd.DataFrame) -> FeatureSet:
        """Crée un ensemble complet de features pour l'IA"""
        try:
            logger.info(f"Création de features pour {len(data)} enregistrements")

            # Assurer l'ordre chronologique
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)
                timestamp_index = pd.to_datetime(data['timestamp'])
            else:
                timestamp_index = pd.date_range(start='2024-01-01', periods=len(data), freq='1min')

            # 1. Features de prix
            price_features = self._create_price_features(data)

            # 2. Features techniques (indicateurs)
            technical_features = self._create_technical_features(data)

            # 3. Features de volume
            volume_features = self._create_volume_features(data)

            # 4. Features temporelles
            time_features = self._create_time_features(timestamp_index)

            # 5. Features de marché
            market_features = self._create_market_features(data)

            # 6. Variable cible (prédiction)
            target_values = self._create_target_variable(data)

            # Combiner toutes les features
            all_features = pd.concat([
                price_features,
                technical_features,
                volume_features,
                time_features,
                market_features
            ], axis=1)

            feature_names = all_features.columns.tolist()

            # Synchroniser les index
            min_length = min(len(all_features), len(target_values), len(timestamp_index))

            feature_set = FeatureSet(
                price_features=price_features.iloc[:min_length],
                technical_features=technical_features.iloc[:min_length],
                volume_features=volume_features.iloc[:min_length],
                time_features=time_features.iloc[:min_length],
                market_features=market_features.iloc[:min_length],
                target_values=target_values.iloc[:min_length],
                feature_names=feature_names,
                timestamp_index=timestamp_index[:min_length]
            )

            self.processing_stats['features_created'] += len(feature_names)
            logger.info(f"Features créées: {len(feature_names)} features, {min_length} échantillons")

            return feature_set

        except Exception as e:
            logger.error(f"Erreur lors de la création des features: {e}")
            # Retourner un FeatureSet vide en cas d'erreur
            return FeatureSet(
                price_features=pd.DataFrame(),
                technical_features=pd.DataFrame(),
                volume_features=pd.DataFrame(),
                time_features=pd.DataFrame(),
                market_features=pd.DataFrame(),
                target_values=pd.Series(),
                feature_names=[],
                timestamp_index=pd.DatetimeIndex([])
            )

    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée les features basées sur les prix"""
        features = pd.DataFrame(index=data.index)

        if 'close_price' in data.columns:
            close = data['close_price']

            # Prix bruts normalisés
            features['close'] = close
            if 'open_price' in data.columns:
                features['open'] = data['open_price']
            if 'high_price' in data.columns:
                features['high'] = data['high_price']
            if 'low_price' in data.columns:
                features['low'] = data['low_price']

            # Rendements
            features['return_1'] = close.pct_change(1)
            features['return_5'] = close.pct_change(5)
            features['return_15'] = close.pct_change(15)

            # Log returns
            features['log_return_1'] = np.log(close / close.shift(1))
            features['log_return_5'] = np.log(close / close.shift(5))

            # Volatilité réalisée
            features['volatility_5'] = features['return_1'].rolling(5).std()
            features['volatility_15'] = features['return_1'].rolling(15).std()
            features['volatility_30'] = features['return_1'].rolling(30).std()

            # Body et shadows des chandeliers
            if all(col in data.columns for col in ['open_price', 'high_price', 'low_price']):
                features['body_size'] = abs(close - data['open_price'])
                features['upper_shadow'] = data['high_price'] - np.maximum(close, data['open_price'])
                features['lower_shadow'] = np.minimum(close, data['open_price']) - data['low_price']
                features['total_range'] = data['high_price'] - data['low_price']

                # Ratios
                features['body_to_range'] = features['body_size'] / (features['total_range'] + 1e-8)
                features['upper_shadow_ratio'] = features['upper_shadow'] / (features['total_range'] + 1e-8)
                features['lower_shadow_ratio'] = features['lower_shadow'] / (features['total_range'] + 1e-8)

            # Moyennes mobiles
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = close.rolling(period).mean()
                features[f'price_to_sma_{period}'] = close / features[f'sma_{period}']

            # EMA
            for period in [12, 26]:
                features[f'ema_{period}'] = close.ewm(span=period).mean()
                features[f'price_to_ema_{period}'] = close / features[f'ema_{period}']

        return features

    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée les features d'indicateurs techniques"""
        features = pd.DataFrame(index=data.index)

        if 'close_price' not in data.columns:
            return features

        close = data['close_price']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['bb_upper'] = sma20 + (2 * std20)
        features['bb_lower'] = sma20 - (2 * std20)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_width'] + 1e-8)

        # Momentum
        features['momentum_5'] = close / close.shift(5)
        features['momentum_10'] = close / close.shift(10)
        features['momentum_20'] = close / close.shift(20)

        # Rate of Change
        features['roc_5'] = ((close - close.shift(5)) / close.shift(5)) * 100
        features['roc_10'] = ((close - close.shift(10)) / close.shift(10)) * 100

        return features

    def _create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée les features basées sur le volume"""
        features = pd.DataFrame(index=data.index)

        if 'volume' not in data.columns:
            # Volume synthétique basé sur la volatilité
            if 'close_price' in data.columns:
                returns = data['close_price'].pct_change()
                features['volume_synthetic'] = abs(returns) * 1000  # Volume synthétique
                features['volume_sma_5'] = features['volume_synthetic'].rolling(5).mean()
                features['volume_ratio'] = features['volume_synthetic'] / (features['volume_sma_5'] + 1e-8)
            return features

        volume = data['volume']

        # Volume brut
        features['volume'] = volume

        # Moyennes mobiles de volume
        features['volume_sma_5'] = volume.rolling(5).mean()
        features['volume_sma_10'] = volume.rolling(10).mean()
        features['volume_sma_20'] = volume.rolling(20).mean()

        # Ratios de volume
        features['volume_ratio_5'] = volume / (features['volume_sma_5'] + 1e-8)
        features['volume_ratio_20'] = volume / (features['volume_sma_20'] + 1e-8)

        # Volume change
        features['volume_change'] = volume.pct_change()
        features['volume_momentum'] = volume / volume.shift(5)

        return features

    def _create_time_features(self, timestamp_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Crée les features temporelles"""
        features = pd.DataFrame(index=range(len(timestamp_index)))

        # Features cycliques
        features['hour'] = timestamp_index.hour
        features['minute'] = timestamp_index.minute
        features['day_of_week'] = timestamp_index.dayofweek
        features['day_of_month'] = timestamp_index.day
        features['month'] = timestamp_index.month

        # Encodage cyclique pour capturer la nature cyclique du temps
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['minute_sin'] = np.sin(2 * np.pi * features['minute'] / 60)
        features['minute_cos'] = np.cos(2 * np.pi * features['minute'] / 60)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        # Sessions de trading (pour contexte)
        features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['is_european_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['is_american_session'] = ((features['hour'] >= 16) & (features['hour'] < 24)).astype(int)

        return features

    def _create_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée les features de conditions de marché"""
        features = pd.DataFrame(index=data.index)

        if 'close_price' not in data.columns:
            return features

        close = data['close_price']

        # Trend strength
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        features['trend_strength'] = (sma20 - sma50) / sma50
        features['is_uptrend'] = (sma20 > sma50).astype(int)

        # Support/Resistance levels (simplified)
        features['resistance_level'] = close.rolling(20).max()
        features['support_level'] = close.rolling(20).min()
        features['distance_to_resistance'] = (features['resistance_level'] - close) / close
        features['distance_to_support'] = (close - features['support_level']) / close

        # Market volatility regime
        volatility = close.pct_change().rolling(20).std()
        vol_percentile = volatility.rolling(100).rank(pct=True)
        features['vol_regime_low'] = (vol_percentile < 0.33).astype(int)
        features['vol_regime_medium'] = ((vol_percentile >= 0.33) & (vol_percentile < 0.66)).astype(int)
        features['vol_regime_high'] = (vol_percentile >= 0.66).astype(int)

        return features

    def _create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """Crée la variable cible pour la prédiction"""
        if 'close_price' not in data.columns:
            return pd.Series(index=data.index)

        close = data['close_price']

        # Prédiction du mouvement futur (classification)
        future_return = close.shift(-self.prediction_horizon).pct_change(self.prediction_horizon)

        # Seuils pour définir les mouvements significatifs
        threshold = future_return.std() * 0.5  # 0.5 écart-type

        # Classification: 0=baisse, 1=stable, 2=hausse
        target = pd.Series(1, index=data.index)  # Défaut: stable
        target[future_return < -threshold] = 0  # Baisse
        target[future_return > threshold] = 2  # Hausse

        return target

    def normalize_features(self, feature_set: FeatureSet, fit_scalers: bool = True) -> FeatureSet:
        """Normalise les features pour l'IA"""
        try:
            normalized_set = FeatureSet(
                price_features=feature_set.price_features.copy(),
                technical_features=feature_set.technical_features.copy(),
                volume_features=feature_set.volume_features.copy(),
                time_features=feature_set.time_features.copy(),
                market_features=feature_set.market_features.copy(),
                target_values=feature_set.target_values.copy(),
                feature_names=feature_set.feature_names.copy(),
                timestamp_index=feature_set.timestamp_index.copy()
            )

            # Normaliser les features de prix
            if not normalized_set.price_features.empty:
                if fit_scalers:
                    normalized_set.price_features = pd.DataFrame(
                        self.scaler_price.fit_transform(normalized_set.price_features.fillna(0)),
                        columns=normalized_set.price_features.columns,
                        index=normalized_set.price_features.index
                    )
                else:
                    normalized_set.price_features = pd.DataFrame(
                        self.scaler_price.transform(normalized_set.price_features.fillna(0)),
                        columns=normalized_set.price_features.columns,
                        index=normalized_set.price_features.index
                    )

            # Normaliser les features techniques
            if not normalized_set.technical_features.empty:
                if fit_scalers:
                    normalized_set.technical_features = pd.DataFrame(
                        self.scaler_technical.fit_transform(normalized_set.technical_features.fillna(0)),
                        columns=normalized_set.technical_features.columns,
                        index=normalized_set.technical_features.index
                    )
                else:
                    normalized_set.technical_features = pd.DataFrame(
                        self.scaler_technical.transform(normalized_set.technical_features.fillna(0)),
                        columns=normalized_set.technical_features.columns,
                        index=normalized_set.technical_features.index
                    )

            # Normaliser les features de volume
            if not normalized_set.volume_features.empty:
                if fit_scalers:
                    normalized_set.volume_features = pd.DataFrame(
                        self.scaler_volume.fit_transform(normalized_set.volume_features.fillna(0)),
                        columns=normalized_set.volume_features.columns,
                        index=normalized_set.volume_features.index
                    )
                else:
                    normalized_set.volume_features = pd.DataFrame(
                        self.scaler_volume.transform(normalized_set.volume_features.fillna(0)),
                        columns=normalized_set.volume_features.columns,
                        index=normalized_set.volume_features.index
                    )

            logger.info("Features normalisées avec succès")
            return normalized_set

        except Exception as e:
            logger.error(f"Erreur lors de la normalisation: {e}")
            return feature_set

    def prepare_training_data(self, symbol: str, timeframe: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> FeatureSet:
        """Prépare les données d'entraînement complètes"""
        try:
            # Récupérer les données depuis la base
            if start_date is None:
                start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now(timezone.utc)

            price_data = db_manager.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date
            )

            if not price_data:
                logger.warning(f"Aucune donnée trouvée pour {symbol} {timeframe}")
                return FeatureSet(
                    price_features=pd.DataFrame(),
                    technical_features=pd.DataFrame(),
                    volume_features=pd.DataFrame(),
                    time_features=pd.DataFrame(),
                    market_features=pd.DataFrame(),
                    target_values=pd.Series(),
                    feature_names=[],
                    timestamp_index=pd.DatetimeIndex([])
                )

            # Convertir en DataFrame
            df = pd.DataFrame([data.to_dict() for data in price_data])

            # Valider la qualité des données
            quality_report = self.validate_raw_data(df)
            logger.info(f"Qualité des données: {quality_report.quality_level.value} "
                        f"(score: {quality_report.overall_score:.2f})")

            if quality_report.overall_score < 0.3:
                logger.error("Qualité des données trop faible pour l'entraînement")
                return FeatureSet(
                    price_features=pd.DataFrame(),
                    technical_features=pd.DataFrame(),
                    volume_features=pd.DataFrame(),
                    time_features=pd.DataFrame(),
                    market_features=pd.DataFrame(),
                    target_values=pd.Series(),
                    feature_names=[],
                    timestamp_index=pd.DatetimeIndex([])
                )

            # Nettoyer les données
            cleaned_df = self.clean_price_data(df)

            # Créer les features
            feature_set = self.create_features(cleaned_df)

            # Normaliser
            normalized_features = self.normalize_features(feature_set, fit_scalers=True)

            self.processing_stats['last_processing_time'] = datetime.now(timezone.utc)

            return normalized_features

        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données d'entraînement: {e}")
            return FeatureSet(
                price_features=pd.DataFrame(),
                technical_features=pd.DataFrame(),
                volume_features=pd.DataFrame(),
                time_features=pd.DataFrame(),
                market_features=pd.DataFrame(),
                target_values=pd.Series(),
                feature_names=[],
                timestamp_index=pd.DatetimeIndex([])
            )

    def get_processing_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de traitement"""
        return self.processing_stats.copy()


# Instance globale
preprocessor = DataPreprocessor()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def validate_data_quality(data: pd.DataFrame) -> DataQualityReport:
    """Valide la qualité des données"""
    return preprocessor.validate_raw_data(data)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données"""
    return preprocessor.clean_price_data(data)


def create_features(data: pd.DataFrame) -> FeatureSet:
    """Crée les features"""
    return preprocessor.create_features(data)


def prepare_training_data(symbol: str, timeframe: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> FeatureSet:
    """Prépare les données d'entraînement"""
    return preprocessor.prepare_training_data(symbol, timeframe, start_date, end_date)


if __name__ == "__main__":
    # Test du préprocesseur
    print("🧹 Test du préprocesseur de données...")

    try:
        # Créer des données de test
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'open_price': np.random.normal(100, 5, 100),
            'high_price': np.random.normal(102, 5, 100),
            'low_price': np.random.normal(98, 5, 100),
            'close_price': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        })

        # Ajouter quelques valeurs aberrantes
        test_data.loc[10, 'close_price'] = 1000  # Outlier
        test_data.loc[20, 'high_price'] = np.nan  # Valeur manquante
        test_data.loc[30, 'low_price'] = test_data.loc[30, 'high_price'] + 10  # Incohérence

        print(f"📊 Données de test créées: {len(test_data)} enregistrements")

        # Validation de la qualité
        quality_report = validate_data_quality(test_data)
        print(f"📈 Qualité initiale: {quality_report.quality_level.value} (score: {quality_report.overall_score:.2f})")
        print(f"   Issues: {len(quality_report.issues)}")
        for issue in quality_report.issues[:3]:  # Afficher les 3 premiers
            print(f"   - {issue}")

        # Nettoyage
        cleaned_data = clean_data(test_data)
        print(f"🧽 Données nettoyées: {len(cleaned_data)} enregistrements")

        # Création de features
        feature_set = create_features(cleaned_data)
        print(f"🎯 Features créées: {len(feature_set.feature_names)} features")
        print(f"   Features de prix: {feature_set.price_features.shape[1]}")
        print(f"   Features techniques: {feature_set.technical_features.shape[1]}")
        print(f"   Features de volume: {feature_set.volume_features.shape[1]}")
        print(f"   Features temporelles: {feature_set.time_features.shape[1]}")

        # Statistiques
        stats = preprocessor.get_processing_stats()
        print(f"\n📊 Statistiques de traitement:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("✅ Test du préprocesseur réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test du préprocesseur échoué: {e}")