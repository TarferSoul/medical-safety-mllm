#!/usr/bin/env python3
"""
Example script showing how to use the configuration system.
这个示例展示如何在脚本中使用配置文件。
"""

import sys
import argparse
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from config_loader import load_config, create_openai_client


def example_prediction_script():
    """Example: How to use config in a prediction script."""
    print("\n" + "="*60)
    print("Example 1: Prediction Script with Config")
    print("="*60)

    # Load configuration
    config = load_config()

    # Get API configuration (automatically formats URLs and handles auth mode)
    print("\n--- Basic Auth Mode (default) ---")
    api_config = config.get_prediction_api_config()

    print(f"Prediction API Settings:")
    print(f"  Auth mode: {api_config['auth_mode']}")
    print(f"  Base URL: {api_config['base_url']}")
    print(f"  Model: {api_config['model']}")
    if api_config['auth_mode'] == 'basic':
        print(f"  API AK: {api_config['api_ak'][:20]}...")
    else:
        print(f"  API Key: {api_config['api_key'][:20]}...")

    # Get with Direct API Key mode
    print("\n--- Direct API Key Mode ---")
    api_config_direct = config.get_prediction_api_config(use_auth=False)

    print(f"Prediction API Settings:")
    print(f"  Auth mode: {api_config_direct['auth_mode']}")
    print(f"  Base URL: {api_config_direct['base_url']}")
    print(f"  Model: {api_config_direct['model']}")
    print(f"  API Key: {api_config_direct['api_key'][:20]}...")

    # Get evaluation settings
    eval_config = config.get_evaluation_config()

    print(f"\nEvaluation Settings:")
    print(f"  Concurrency: {eval_config['prediction_concurrency']}")
    print(f"  Max retries: {eval_config['prediction_max_retries']}")
    print(f"  Max tokens: {eval_config['prediction_max_tokens']}")
    print(f"  Temperature: {eval_config['prediction_temperature']}")

    # Get data paths
    paths = config.get_data_paths()

    print(f"\nData Paths:")
    print(f"  Test data: {paths['test_data']}")
    print(f"  Output dir: {paths['output_dir']}")


def example_with_cli_override():
    """Example: CLI arguments can override config values."""
    print("\n" + "="*60)
    print("Example 2: CLI Override Config")
    print("="*60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency")
    parser.add_argument("--max_samples", type=int, default=None, help="Override max samples")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get config values with CLI overrides
    model = args.model if args.model else config.get("prediction_api.model")
    concurrency = args.concurrency if args.concurrency else config.get("evaluation.prediction_concurrency")
    max_samples = args.max_samples if args.max_samples else config.get("evaluation.default_max_samples")

    print(f"\nFinal Settings (after CLI override):")
    print(f"  Model: {model}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Max samples: {max_samples}")


def example_judge_script():
    """Example: How to use config in a judge script."""
    print("\n" + "="*60)
    print("Example 3: Judge Script with Config")
    print("="*60)

    config = load_config()

    # Get judge API config
    judge_config = config.get_judge_api_config()

    print(f"\nJudge API Settings:")
    print(f"  Base URL: {judge_config['base_url']}")
    print(f"  Model: {judge_config['model']}")

    # Get judge prompt template
    judge_prompt = config.get_judge_prompt()

    print(f"\nJudge Prompt Template:")
    print(f"  Length: {len(judge_prompt)} characters")
    print(f"  First 100 chars: {judge_prompt[:100]}...")

    # Get judge-specific settings
    judge_settings = {
        "max_tokens": config.get("evaluation.judge_max_tokens"),
        "temperature": config.get("evaluation.judge_temperature"),
        "max_retries": config.get("evaluation.judge_max_retries"),
        "concurrency": config.get("evaluation.judge_concurrency"),
    }

    print(f"\nJudge Settings:")
    for key, value in judge_settings.items():
        print(f"  {key}: {value}")


def example_reasoning_script():
    """Example: How to use config in a reasoning script."""
    print("\n" + "="*60)
    print("Example 4: Reasoning Script with Config")
    print("="*60)

    config = load_config()

    # Get reasoning API config (can choose auth mode)
    print("\n--- Basic Auth Mode ---")
    reasoning_config = config.get_reasoning_api_config(use_auth=True)

    print(f"  Base URL: {reasoning_config['base_url']}")
    print(f"  Model: {reasoning_config['model']}")
    print(f"  Auth mode: {reasoning_config['auth_mode']}")

    print("\n--- Direct API Key Mode ---")
    reasoning_config = config.get_reasoning_api_config(use_auth=False)

    print(f"  Base URL: {reasoning_config['base_url']}")
    print(f"  Model: {reasoning_config['model']}")
    print(f"  Auth mode: {reasoning_config['auth_mode']}")

    # Get reasoning prompt
    reasoning_prompt = config.get_reasoning_prompt()

    print(f"\nReasoning Prompt Template:")
    print(f"  Length: {len(reasoning_prompt)} characters")
    print(f"  First 100 chars: {reasoning_prompt[:100]}...")


def example_normalize_script():
    """Example: How to use config in a normalize script."""
    print("\n" + "="*60)
    print("Example 4.5: Normalize Script with Config")
    print("="*60)

    config = load_config()

    # Get normalize API config (can choose auth mode)
    print("\n--- Basic Auth Mode ---")
    normalize_config = config.get_normalize_api_config(use_auth=True)

    print(f"  Base URL: {normalize_config['base_url']}")
    print(f"  Model: {normalize_config['model']}")
    print(f"  Auth mode: {normalize_config['auth_mode']}")

    print("\n--- Direct API Key Mode ---")
    normalize_config = config.get_normalize_api_config(use_auth=False)

    print(f"  Base URL: {normalize_config['base_url']}")
    print(f"  Model: {normalize_config['model']}")
    print(f"  Auth mode: {normalize_config['auth_mode']}")


def example_data_processing():
    """Example: How to use config in data processing scripts."""
    print("\n" + "="*60)
    print("Example 5: Data Processing with Config")
    print("="*60)

    config = load_config()

    # Get data processing settings
    data_config = config.get_data_processing_config()

    print(f"\nData Processing Settings:")
    print(f"  Instruction: {data_config['default_instruction']}")
    print(f"  Test samples: {data_config['test_samples']}")
    print(f"  Random seed: {data_config['random_seed']}")

    # Get paths
    paths = config.get_data_paths()

    print(f"\nData Paths:")
    print(f"  MIMIC images: {paths['mimic_images_dir']}")
    print(f"  MIMIC reports: {paths['mimic_reports_dir']}")
    print(f"  Dataset dir: {paths['dataset_dir']}")


def example_dot_notation():
    """Example: Using dot notation for deep access."""
    print("\n" + "="*60)
    print("Example 6: Dot Notation Access")
    print("="*60)

    config = load_config()

    print(f"\nDirect access with dot notation:")
    print(f"  config.get('prediction_api.model_auth'): {config.get('prediction_api.model_auth')}")
    print(f"  config.get('judge_api.port'): {config.get('judge_api.port')}")
    print(f"  config.get('evaluation.prediction_concurrency'): {config.get('evaluation.prediction_concurrency')}")
    print(f"  config.get('data_processing.test_samples'): {config.get('data_processing.test_samples')}")

    # With default values
    print(f"\n  config.get('nonexistent.key', 'default'): {config.get('nonexistent.key', 'default')}")


def example_create_client():
    """Example: Creating OpenAI client with unified interface."""
    print("\n" + "="*60)
    print("Example 7: Creating OpenAI Client")
    print("="*60)

    config = load_config()

    # Example 1: Create prediction client with Basic Auth
    print("\n--- Creating Prediction Client (Basic Auth) ---")
    pred_config = config.get_prediction_api_config(use_auth=True)
    print(f"  Auth mode: {pred_config['auth_mode']}")
    print(f"  Model: {pred_config['model']}")
    try:
        pred_client = create_openai_client(pred_config)
        print(f"  ✓ Client created successfully")
    except ImportError as e:
        print(f"  ⚠ {e}")

    # Example 2: Create prediction client with Direct API Key
    print("\n--- Creating Prediction Client (Direct API Key) ---")
    pred_config_direct = config.get_prediction_api_config(use_auth=False)
    print(f"  Auth mode: {pred_config_direct['auth_mode']}")
    print(f"  Model: {pred_config_direct['model']}")
    try:
        pred_client_direct = create_openai_client(pred_config_direct)
        print(f"  ✓ Client created successfully")
    except ImportError as e:
        print(f"  ⚠ {e}")

    # Example 3: Create judge client
    print("\n--- Creating Judge Client ---")
    judge_config = config.get_judge_api_config()
    print(f"  Auth mode: {judge_config['auth_mode']}")
    print(f"  Model: {judge_config['model']}")
    try:
        judge_client = create_openai_client(judge_config)
        print(f"  ✓ Client created successfully")
    except ImportError as e:
        print(f"  ⚠ {e}")

    print("\n  Usage example:")
    print("    response = client.chat.completions.create(")
    print("        model=api_config['model'],")
    print("        messages=[...],")
    print("        max_tokens=4096")
    print("    )")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Configuration System Examples")
    print("="*60)

    try:
        # Run all examples
        example_prediction_script()
        example_judge_script()
        example_reasoning_script()
        example_normalize_script()
        example_data_processing()
        example_dot_notation()
        example_create_client()

        # This one needs CLI args, so we show it last
        # example_with_cli_override()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure config.yaml exists in the project root directory.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
