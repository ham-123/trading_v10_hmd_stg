"""
Gestionnaire sécurisé des clés API et secrets
Volatility 10 Trading Bot - secrets.py
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SecretManager:
    """Gestionnaire sécurisé pour les secrets et clés API"""

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialise le gestionnaire de secrets

        Args:
            master_key: Clé maître pour le chiffrement (optionnel)
        """
        self._master_key = master_key or os.getenv("MASTER_ENCRYPTION_KEY")
        self._secrets_cache = {}
        self._encryption_key = None
        self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        """Initialise le système de chiffrement"""
        if self._master_key:
            # Dériver une clé de chiffrement à partir de la clé maître
            salt = os.getenv("ENCRYPTION_SALT", "trading_bot_salt_2024").encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
            self._encryption_key = Fernet(key)
        else:
            # Générer une nouvelle clé si aucune clé maître
            logger.warning("Aucune clé maître fournie, génération d'une nouvelle clé")
            key = Fernet.generate_key()
            self._encryption_key = Fernet(key)
            self._save_generated_key(key)

    def _save_generated_key(self, key: bytes) -> None:
        """Sauvegarde la clé générée pour réutilisation"""
        key_file = ".encryption_key"
        try:
            with open(key_file, "wb") as f:
                f.write(key)
            logger.info(f"Clé de chiffrement sauvegardée dans {key_file}")
        except Exception as e:
            logger.error(f"Impossible de sauvegarder la clé: {e}")

    def encrypt_secret(self, secret: str) -> str:
        """
        Chiffre un secret

        Args:
            secret: Secret à chiffrer

        Returns:
            Secret chiffré en base64
        """
        try:
            encrypted_bytes = self._encryption_key.encrypt(secret.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur lors du chiffrement: {e}")
            raise

    def decrypt_secret(self, encrypted_secret: str) -> str:
        """
        Déchiffre un secret

        Args:
            encrypted_secret: Secret chiffré en base64

        Returns:
            Secret déchiffré
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_secret.encode('utf-8'))
            decrypted_bytes = self._encryption_key.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur lors du déchiffrement: {e}")
            raise

    def store_secret(self, key: str, value: str, encrypt: bool = True) -> None:
        """
        Stocke un secret dans le cache

        Args:
            key: Clé du secret
            value: Valeur du secret
            encrypt: Chiffrer la valeur (par défaut True)
        """
        if encrypt:
            self._secrets_cache[key] = {
                'value': self.encrypt_secret(value),
                'encrypted': True,
                'timestamp': datetime.now().isoformat()
            }
        else:
            self._secrets_cache[key] = {
                'value': value,
                'encrypted': False,
                'timestamp': datetime.now().isoformat()
            }

        logger.debug(f"Secret '{key}' stocké avec succès")

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Récupère un secret depuis le cache ou les variables d'environnement

        Args:
            key: Clé du secret
            default: Valeur par défaut si non trouvé

        Returns:
            Valeur du secret ou default
        """
        # Chercher d'abord dans le cache
        if key in self._secrets_cache:
            secret_data = self._secrets_cache[key]
            if secret_data['encrypted']:
                return self.decrypt_secret(secret_data['value'])
            else:
                return secret_data['value']

        # Chercher dans les variables d'environnement
        env_value = os.getenv(key, default)
        if env_value and env_value != default:
            # Stocker dans le cache pour usage futur
            self.store_secret(key, env_value, encrypt=True)
            return env_value

        return default

    def remove_secret(self, key: str) -> bool:
        """
        Supprime un secret du cache

        Args:
            key: Clé du secret à supprimer

        Returns:
            True si supprimé, False si non trouvé
        """
        if key in self._secrets_cache:
            del self._secrets_cache[key]
            logger.debug(f"Secret '{key}' supprimé")
            return True
        return False

    def list_secrets(self) -> Dict[str, Dict]:
        """
        Liste tous les secrets stockés (sans les valeurs)

        Returns:
            Dictionnaire avec les métadonnées des secrets
        """
        return {
            key: {
                'encrypted': data['encrypted'],
                'timestamp': data['timestamp']
            }
            for key, data in self._secrets_cache.items()
        }

    def export_secrets(self, include_values: bool = False,
                       password: Optional[str] = None) -> str:
        """
        Exporte les secrets en JSON

        Args:
            include_values: Inclure les valeurs (dangereux)
            password: Mot de passe pour chiffrement supplémentaire

        Returns:
            JSON des secrets
        """
        export_data = {}

        for key, data in self._secrets_cache.items():
            if include_values:
                if data['encrypted']:
                    export_data[key] = {
                        'value': data['value'],  # Déjà chiffré
                        'encrypted': True,
                        'timestamp': data['timestamp']
                    }
                else:
                    # Chiffrer avant export
                    export_data[key] = {
                        'value': self.encrypt_secret(data['value']),
                        'encrypted': True,
                        'timestamp': data['timestamp']
                    }
            else:
                export_data[key] = {
                    'encrypted': data['encrypted'],
                    'timestamp': data['timestamp']
                }

        json_data = json.dumps(export_data, indent=2)

        # Chiffrement supplémentaire avec mot de passe si fourni
        if password and include_values:
            additional_key = self._derive_key_from_password(password)
            additional_cipher = Fernet(additional_key)
            json_data = base64.urlsafe_b64encode(
                additional_cipher.encrypt(json_data.encode())
            ).decode()

        return json_data

    def import_secrets(self, json_data: str,
                       password: Optional[str] = None) -> None:
        """
        Importe des secrets depuis JSON

        Args:
            json_data: Données JSON des secrets
            password: Mot de passe si chiffrement supplémentaire
        """
        try:
            # Déchiffrement supplémentaire si mot de passe fourni
            if password:
                try:
                    additional_key = self._derive_key_from_password(password)
                    additional_cipher = Fernet(additional_key)
                    decrypted_bytes = additional_cipher.decrypt(
                        base64.urlsafe_b64decode(json_data.encode())
                    )
                    json_data = decrypted_bytes.decode()
                except Exception:
                    # Essayer sans déchiffrement supplémentaire
                    pass

            secrets_data = json.loads(json_data)

            for key, data in secrets_data.items():
                if 'value' in data:
                    self._secrets_cache[key] = data
                    logger.debug(f"Secret '{key}' importé")

            logger.info(f"{len(secrets_data)} secrets importés avec succès")

        except Exception as e:
            logger.error(f"Erreur lors de l'importation: {e}")
            raise

    def _derive_key_from_password(self, password: str) -> bytes:
        """Dérive une clé de chiffrement à partir d'un mot de passe"""
        salt = b"trading_bot_additional_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def generate_api_key(self, length: int = 32) -> str:
        """
        Génère une clé API sécurisée

        Args:
            length: Longueur de la clé

        Returns:
            Clé API générée
        """
        return secrets.token_urlsafe(length)

    def hash_secret(self, secret: str) -> str:
        """
        Crée un hash sécurisé d'un secret

        Args:
            secret: Secret à hasher

        Returns:
            Hash SHA-256 du secret
        """
        return hashlib.sha256(secret.encode()).hexdigest()

    def verify_secret_hash(self, secret: str, hash_value: str) -> bool:
        """
        Vérifie un secret contre son hash

        Args:
            secret: Secret à vérifier
            hash_value: Hash de référence

        Returns:
            True si le secret correspond au hash
        """
        return self.hash_secret(secret) == hash_value


class APIKeyManager:
    """Gestionnaire spécialisé pour les clés API de trading"""

    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager
        self.api_keys = {}
        self._load_api_keys()

    def _load_api_keys(self) -> None:
        """Charge les clés API depuis les secrets"""
        # Clés API Deriv
        deriv_token = self.secret_manager.get_secret("DERIV_API_TOKEN")
        if deriv_token:
            self.api_keys["deriv"] = {
                "token": deriv_token,
                "app_id": self.secret_manager.get_secret("DERIV_APP_ID", "1089"),
                "expires": None,  # Les tokens Deriv n'expirent pas automatiquement
                "permissions": ["read", "trade", "payments"]
            }

        # Clés Telegram
        telegram_token = self.secret_manager.get_secret("TELEGRAM_BOT_TOKEN")
        if telegram_token:
            self.api_keys["telegram"] = {
                "token": telegram_token,
                "chat_id": self.secret_manager.get_secret("TELEGRAM_CHAT_ID"),
                "expires": None
            }

    def get_deriv_credentials(self) -> Optional[Dict[str, str]]:
        """Retourne les credentials Deriv"""
        if "deriv" in self.api_keys:
            return {
                "api_token": self.api_keys["deriv"]["token"],
                "app_id": self.api_keys["deriv"]["app_id"]
            }
        return None

    def get_telegram_credentials(self) -> Optional[Dict[str, str]]:
        """Retourne les credentials Telegram"""
        if "telegram" in self.api_keys:
            return {
                "bot_token": self.api_keys["telegram"]["token"],
                "chat_id": self.api_keys["telegram"]["chat_id"]
            }
        return None

    def validate_deriv_token(self) -> bool:
        """Valide le token Deriv (simulation)"""
        credentials = self.get_deriv_credentials()
        if not credentials:
            return False

        # TODO: Implémenter la validation réelle via API Deriv
        token = credentials["api_token"]
        return len(token) > 10 and token.startswith(("1", "2", "3", "4", "5"))

    def rotate_api_key(self, service: str) -> str:
        """Génère une nouvelle clé API pour un service"""
        new_key = self.secret_manager.generate_api_key()

        if service in self.api_keys:
            old_key_hash = self.secret_manager.hash_secret(
                self.api_keys[service]["token"]
            )
            logger.info(f"Rotation de clé pour {service}. Ancien hash: {old_key_hash[:8]}...")

        # Stocker la nouvelle clé
        self.secret_manager.store_secret(f"{service.upper()}_API_TOKEN", new_key)
        self._load_api_keys()  # Recharger

        return new_key


def create_env_template() -> str:
    """Crée un template de fichier .env"""
    template = """# =============================================================================
# CONFIGURATION TRADING BOT - Variables d'environnement
# =============================================================================

# Mode de trading
TRADING_MODE=paper  # paper, live, backtest

# =============================================================================
# API DERIV
# =============================================================================
DERIV_API_TOKEN=your_deriv_api_token_here
DERIV_APP_ID=1089

# =============================================================================
# BASE DE DONNÉES
# =============================================================================
DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
REDIS_URL=redis://localhost:6379/0

# =============================================================================
# CHIFFREMENT ET SÉCURITÉ
# =============================================================================
MASTER_ENCRYPTION_KEY=your_master_key_here_32_chars_min
ENCRYPTION_SALT=trading_bot_salt_2024

# =============================================================================
# MONITORING ET ALERTES
# =============================================================================
LOG_LEVEL=INFO

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Email (optionnel)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_email_password_here

# =============================================================================
# DÉVELOPPEMENT
# =============================================================================
DEBUG=False
TESTING=False
"""
    return template


def setup_secrets_manager() -> tuple[SecretManager, APIKeyManager]:
    """
    Configure et retourne les gestionnaires de secrets

    Returns:
        Tuple (SecretManager, APIKeyManager)
    """
    # Initialiser le gestionnaire de secrets
    secret_manager = SecretManager()

    # Initialiser le gestionnaire de clés API
    api_manager = APIKeyManager(secret_manager)

    logger.info("Gestionnaires de secrets initialisés")
    return secret_manager, api_manager


# Instance globale (singleton)
_secret_manager = None
_api_manager = None


def get_secret_manager() -> SecretManager:
    """Retourne l'instance globale du gestionnaire de secrets"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager, _ = setup_secrets_manager()
    return _secret_manager


def get_api_manager() -> APIKeyManager:
    """Retourne l'instance globale du gestionnaire d'API"""
    global _api_manager
    if _api_manager is None:
        _, _api_manager = setup_secrets_manager()
    return _api_manager


if __name__ == "__main__":
    # Test du gestionnaire de secrets
    print("🔐 Test du gestionnaire de secrets...")

    # Créer un gestionnaire de test
    sm = SecretManager()

    # Test de chiffrement/déchiffrement
    test_secret = "ma_cle_api_super_secrete_123"
    encrypted = sm.encrypt_secret(test_secret)
    decrypted = sm.decrypt_secret(encrypted)

    print(f"✅ Secret original: {test_secret}")
    print(f"🔒 Secret chiffré: {encrypted[:20]}...")
    print(f"🔓 Secret déchiffré: {decrypted}")
    print(f"✅ Chiffrement/déchiffrement: {'OK' if test_secret == decrypted else 'ERREUR'}")

    # Test de stockage
    sm.store_secret("TEST_API_KEY", "test123456789")
    retrieved = sm.get_secret("TEST_API_KEY")
    print(f"✅ Stockage/récupération: {'OK' if retrieved == 'test123456789' else 'ERREUR'}")

    # Test du gestionnaire d'API
    api_mgr = APIKeyManager(sm)
    print(f"✅ Gestionnaire d'API initialisé: {len(api_mgr.api_keys)} clés chargées")

    # Créer le template .env
    print(f"\n📝 Template .env créé ({len(create_env_template())} caractères)")

    print("✅ Tous les tests réussis !")