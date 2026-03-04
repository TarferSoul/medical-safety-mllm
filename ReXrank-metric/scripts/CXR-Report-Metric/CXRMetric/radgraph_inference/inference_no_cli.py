"""
RadGraph inference without using allennlp CLI
Uses AllenNLP Python API directly
"""
import json
import torch
from pathlib import Path


def run_inference_no_cli(model_path, data_path, out_path, cuda_device=0):
    """
    Run RadGraph inference using AllenNLP Python API instead of CLI

    Args:
        model_path: Path to the model.tar.gz checkpoint
        data_path: Path to input JSON (temp_dygie_input.json)
        out_path: Path to output JSON (temp_dygie_output.json)
        cuda_device: GPU device ID (-1 for CPU)
    """
    try:
        # Import AllenNLP components
        from allennlp.models.archival import load_archive
        from allennlp.predictors import Predictor

        # Load the model
        print(f"Loading RadGraph model from {model_path}...")
        archive = load_archive(
            model_path,
            cuda_device=cuda_device,
            overrides='{"model.use_ner_scores_for_train": false}'
        )

        # Create predictor
        predictor = Predictor.from_archive(
            archive,
            predictor_name="dygie",
            dataset_reader_to_load="validation"
        )

        # Read input data
        with open(data_path, 'r') as f:
            input_data = [json.loads(line) for line in f]

        # Run predictions
        print(f"Running inference on {len(input_data)} samples...")
        results = []
        for instance in input_data:
            prediction = predictor.predict_json(instance)
            results.append(prediction)

        # Write output
        with open(out_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        print(f"✅ Inference complete: {out_path}")
        return True

    except ImportError as e:
        print(f"❌ AllenNLP not available: {e}")
        print("Falling back to CLI method...")
        return False
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference_with_fallback(model_path, data_path, out_path, cuda_device=0):
    """
    Try Python API first, fall back to CLI if needed
    """
    import os

    # Try Python API first
    success = run_inference_no_cli(model_path, data_path, out_path, cuda_device)

    if not success:
        # Fall back to CLI
        print("Using allennlp CLI as fallback...")
        cmd = f"allennlp predict {model_path} {data_path} " \
              f"--predictor dygie --include-package dygie " \
              f"--use-dataset-reader " \
              f"--output-file {out_path} " \
              f"--cuda-device {cuda_device} " \
              f"--silent"

        return_code = os.system(cmd)
        if return_code != 0:
            raise RuntimeError(f"AllenNLP inference failed with code {return_code}")

    return True
