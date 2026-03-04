#!/usr/bin/env python3
"""
Configuration loader utility for Medical Safety project.
Loads configuration from config.yaml and provides easy access to settings.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for the project."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to config.yaml. If None, searches in default locations.
        """
        if config_path is None:
            # Search for config.yaml in common locations
            possible_paths = [
                Path(__file__).parent.parent / "config.yaml",  # Project root
                Path.cwd() / "config.yaml",  # Current directory
                Path.cwd().parent / "config.yaml",  # Parent directory
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                f"config.yaml not found. Searched in: {[str(p) for p in possible_paths]}"
            )

        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        print(f"✓ Loaded configuration from: {config_path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "prediction_api.model")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get("prediction_api.model")
            config.get("evaluation.prediction_concurrency", 50)
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def _get_api_config(self, api_key: str, use_auth: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get API configuration with support for both auth modes.

        Args:
            api_key: API configuration key (e.g., "prediction_api", "judge_api")
            use_auth: Override auth mode (True=Basic Auth, False=Direct API Key)

        Returns:
            API configuration dict
        """
        config = self.get(api_key, {})

        # Determine auth mode
        if use_auth is None:
            use_auth = config.get("auth_mode", True)

        # Select appropriate settings
        if use_auth:
            # Basic Auth mode
            result = {
                "base_url": config.get("base_url_auth", ""),
                "api_ak": config.get("api_ak"),
                "api_sk": config.get("api_sk"),
                "model": config.get("model_auth"),
                "auth_mode": "basic"
            }
            # Build full URL
            if "{prefix}" in result["base_url"]:
                result["base_url"] = result["base_url"].format(
                    prefix=config.get("prefix", ""),
                    port=config.get("port", "")
                )
        else:
            # Direct API Key mode
            result = {
                "base_url": config.get("base_url_direct"),
                "api_key": config.get("api_key_direct"),
                "model": config.get("model_direct"),
                "auth_mode": "direct"
            }

        return result

    def get_prediction_api_config(self, use_auth: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get prediction API configuration from evaluation section.

        Args:
            use_auth: Override auth mode (True=Basic Auth, False=Direct API Key)
        """
        return self._get_api_config("evaluation.prediction_api", use_auth)

    def get_judge_api_config(self, use_auth: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get judge API configuration from evaluation section.

        Args:
            use_auth: Override auth mode (True=Basic Auth, False=Direct API Key)
        """
        return self._get_api_config("evaluation.judge_api", use_auth)

    def get_reasoning_api_config(self, use_auth: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get reasoning API configuration.

        Args:
            use_auth: Override auth mode (True=Basic Auth, False=Direct API Key)
        """
        return self._get_api_config("reasoning_api", use_auth)

    def get_normalize_api_config(self, use_auth: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get normalize API configuration.

        Args:
            use_auth: Override auth mode (True=Basic Auth, False=Direct API Key)
        """
        return self._get_api_config("normalization", use_auth)

    def get_data_paths(self) -> Dict[str, str]:
        """Get all data paths configuration."""
        return self.get("data_paths", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get("evaluation", {})

    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.get("data_processing", {})

    def get_normalization_config(self) -> Dict[str, Any]:
        """Get normalization configuration."""
        return self.get("normalization", {})

    def get_judge_prompt(self) -> str:
        """Get judge prompt template."""
        return self.get("judge_prompt.template", "")

    def get_reasoning_prompt(self) -> str:
        """Get reasoning prompt template."""
        return self.get("reasoning_prompt.template", "")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Singleton instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, uses default locations.

    Returns:
        Config instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance

    Raises:
        RuntimeError: If config not loaded yet
    """
    global _global_config
    if _global_config is None:
        # Auto-load with default path
        _global_config = Config()
    return _global_config


def create_openai_client(api_config: Dict[str, Any]):
    """
    Create OpenAI client from API configuration.

    Args:
        api_config: API configuration dict (from get_*_api_config())

    Returns:
        OpenAI client instance

    Example:
        config = load_config()
        api_config = config.get_prediction_api_config()
        client = create_openai_client(api_config)
    """
    try:
        import openai
        import base64
    except ImportError:
        raise ImportError("openai library not installed. Please run 'pip install openai'")

    auth_mode = api_config.get("auth_mode", "basic")

    if auth_mode == "basic":
        # Basic Auth with AK/SK
        auth_string = f"{api_config['api_ak']}:{api_config['api_sk']}"
        b64_auth = base64.b64encode(auth_string.encode()).decode()

        client = openai.OpenAI(
            base_url=api_config['base_url'],
            api_key=b64_auth,
            default_headers={"Authorization": f"Basic {b64_auth}"}
        )
    else:
        # Direct API Key
        client = openai.OpenAI(
            base_url=api_config['base_url'],
            api_key=api_config['api_key']
        )

    return client


# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config()

    # Access configurations
    print("\n=== Prediction API Config ===")
    pred_config = config.get_prediction_api_config()
    print(f"Base URL: {pred_config['base_url']}")
    print(f"Model: {pred_config['model']}")

    print("\n=== Judge API Config ===")
    judge_config = config.get_judge_api_config()
    print(f"Base URL: {judge_config['base_url']}")
    print(f"Model: {judge_config['model']}")

    print("\n=== Data Paths ===")
    paths = config.get_data_paths()
    print(f"Test data: {paths['test_data']}")
    print(f"Output dir: {paths['output_dir']}")

    print("\n=== Evaluation Settings ===")
    eval_config = config.get_evaluation_config()
    print(f"Prediction concurrency: {eval_config['prediction_concurrency']}")
    print(f"Judge concurrency: {eval_config['judge_concurrency']}")

    print("\n=== Using dot notation ===")
    print(f"Test samples: {config.get('data_processing.test_samples')}")
    print(f"Random seed: {config.get('data_processing.random_seed')}")
