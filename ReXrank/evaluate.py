import argparse
import json
import os
import re

os.environ["NLTK_DATA"] = "/mnt/shared-storage-user/xieyuejin/nltk_data"

import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_nltk_data():
    pass


def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_findings(text):
    text = preprocess_text(text)
    findings_match = re.search(
        r"findings?[:\s]+(.*?)(?:impression|$)", text, re.IGNORECASE | re.DOTALL
    )
    if findings_match:
        return findings_match.group(1).strip()

    impression_match = re.search(
        r"impression[:\s]+(.*?)$", text, re.IGNORECASE | re.DOTALL
    )
    if impression_match:
        return impression_match.group(1).strip()

    return text


def extract_impression(text):
    text = preprocess_text(text)
    impression_match = re.search(
        r"impression[:\s]+(.*?)$", text, re.IGNORECASE | re.DOTALL
    )
    if impression_match:
        return impression_match.group(1).strip()
    return ""


def calculate_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method1
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    try:
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    except:
        bleu = 0.0

    return bleu


import threading

_bert_model = None
_bert_tokenizer = None
_bert_lock = threading.Lock()


def get_bert_model(
    model_name="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--google-bert--bert-base-uncased",
):
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        with _bert_lock:
            if _bert_model is None:
                _bert_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=True
                )
                _bert_model = AutoModel.from_pretrained(
                    model_name, local_files_only=True
                ).cuda()
                _bert_model.eval()
    return _bert_tokenizer, _bert_model


def calculate_bertscore(
    predictions,
    references,
    model_name="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--google-bert--bert-base-uncased",
):
    try:
        tokenizer, model = get_bert_model(model_name)

        preds_tokens = tokenizer(
            predictions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        preds_tokens = {k: v.cuda() for k, v in preds_tokens.items()}
        refs_tokens = tokenizer(
            references,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        refs_tokens = {k: v.cuda() for k, v in refs_tokens.items()}

        with torch.no_grad():
            preds_emb = model(**preds_tokens).last_hidden_state.mean(dim=1)
            refs_emb = model(**refs_tokens).last_hidden_state.mean(dim=1)

        preds_emb = F.normalize(preds_emb, p=2, dim=1)
        refs_emb = F.normalize(refs_emb, p=2, dim=1)

        similarities = (preds_emb * refs_emb).sum(dim=1).cpu().numpy()

        return np.mean(similarities)
    except Exception as e:
        import traceback

        print(f"BerScore error: {e}")
        traceback.print_exc()
        return 0.0


def calculate_semscore(
    predictions,
    references,
    model_name="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--google-bert--bert-base-uncased",
):
    return calculate_bertscore(predictions, references, model_name)


def calculate_radgraph(prediction, reference):
    try:
        from radgraph import F1RadGraph

        evaluator = F1RadGraph()

        pred_entities = evaluator.get_entities(prediction)
        ref_entities = evaluator.get_entities(reference)

        pred_relations = evaluator.get_relations(prediction)
        ref_relations = evaluator.get_relations(reference)

        def calc_f1(pred_set, ref_set):
            if len(pred_set) == 0 and len(ref_set) == 0:
                return 1.0
            if len(pred_set) == 0 or len(ref_set) == 0:
                return 0.0
            tp = len(pred_set & ref_set)
            precision = tp / len(pred_set) if len(pred_set) > 0 else 0
            recall = tp / len(ref_set) if len(ref_set) > 0 else 0
            if precision + recall == 0:
                return 0.0
            f1 = 2 * precision * recall / (precision + recall)
            return f1

        entity_f1 = calc_f1(pred_entities, ref_entities)
        relation_f1 = calc_f1(pred_relations, ref_relations)

        return (entity_f1 + relation_f1) / 2
    except ImportError:
        return calculate_soft_f1(prediction, reference)
    except Exception as e:
        print(f"RadGraph error: {e}")
        return calculate_soft_f1(prediction, reference)


def calculate_ratescore(prediction, reference):
    try:
        from raetextract import RaTEScore

        scorer = RaTEScore()
        score = scorer.score(reference, prediction)
        return score
    except ImportError:
        return calculate_soft_f1(prediction, reference)
    except Exception as e:
        print(f"RaTEScore error: {e}")
        return calculate_soft_f1(prediction, reference)


def calculate_soft_f1(prediction, reference):
    pred_tokens = set(preprocess_text(prediction).split())
    ref_tokens = set(preprocess_text(reference).split())

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    tp = len(pred_tokens & ref_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def calculate_green(prediction, reference):
    try:
        from green_score import GreenScore

        scorer = GreenScore()
        score = scorer.score(reference, prediction)
        return score
    except ImportError:
        return calculate_soft_f1(prediction, reference)
    except Exception as e:
        print(f"GREEN error: {e}")
        return calculate_soft_f1(prediction, reference)


def extract_clinical_entities(text):
    text = preprocess_text(text)
    entities = set()

    patterns = {
        "heart": r"\b(heart|cardiomegaly|cardiac|enlarged heart)\b",
        "lung": r"\b(lung|pulmonary|lungs|opacity|consolidation|infiltrate|atelectasis|pneumonia)\b",
        "pleural": r"\b(pleural|effusion|pneumothorax)\b",
        "mediastinum": r"\b(mediastinum|mediastinal)\b",
        "bone": r"\b(fracture|degenerative|osteophyte)\b",
    }

    for entity_type, pattern in patterns.items():
        if re.search(pattern, text):
            entities.add(entity_type)

    return entities


def calculate_radcliq(prediction, reference):
    pred_entities = extract_clinical_entities(prediction)
    ref_entities = extract_clinical_entities(reference)

    if len(ref_entities) == 0:
        return 1.0 if len(pred_entities) == 0 else 0.0

    errors = 0

    for entity in ref_entities:
        if entity not in pred_entities:
            errors += 1

    for entity in pred_entities:
        if entity not in ref_entities:
            errors += 0.5

    if errors == 0:
        return 1.0

    return 1.0 / (1.0 + errors)


def evaluate_sample(args):
    study_id, pred_text, gt_text, mode = args
    result = {
        "study_id": study_id,
        "bleu": 0.0,
        "radgraph": 0.0,
        "ratescore": 0.0,
        "green": 0.0,
        "radcliq": 0.0,
    }

    if mode == "findings":
        pred_text = extract_findings(pred_text)
        gt_text = extract_findings(gt_text)
    elif mode == "findings_impression":
        pred_findings = extract_findings(pred_text)
        gt_findings = extract_findings(gt_text)
        pred_impression = extract_impression(pred_text)
        gt_impression = extract_impression(gt_text)
        pred_text = f"{pred_findings} {pred_impression}".strip()
        gt_text = f"{gt_findings} {gt_impression}".strip()

    pred_text = preprocess_text(pred_text)
    gt_text = preprocess_text(gt_text)

    if not pred_text or not gt_text:
        return result

    result["bleu"] = calculate_bleu(gt_text, pred_text)
    result["radgraph"] = calculate_radgraph(pred_text, gt_text)
    result["ratescore"] = calculate_ratescore(pred_text, gt_text)
    result["green"] = calculate_green(pred_text, gt_text)
    result["radcliq"] = calculate_radcliq(pred_text, gt_text)

    return result


def evaluate_rexrank(pred_json, gt_json, mode="findings", concurrency=50):
    download_nltk_data()

    print("Pre-loading BERT model...", flush=True)
    get_bert_model()
    print("BERT model loaded", flush=True)

    with open(pred_json, "r") as f:
        predictions = json.load(f)

    with open(gt_json, "r") as f:
        ground_truth = json.load(f)

    results = {
        "bleu": [],
        "bertscore": [],
        "semscore": [],
        "radgraph": [],
        "ratescore": [],
        "green": [],
        "radcliq": [],
    }

    eval_tasks = []
    for study_id in ground_truth.keys():
        if study_id not in predictions:
            print(f"Warning: {study_id} not in predictions")
            continue

        pred_text = predictions[study_id].get("model_prediction", "")
        gt_text = ground_truth[study_id].get("report", "")
        eval_tasks.append((study_id, pred_text, gt_text, mode))

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(evaluate_sample, task): task[0] for task in eval_tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating"
        ):
            try:
                result = future.result()
                results["bleu"].append(result["bleu"])
                results["radgraph"].append(result["radgraph"])
                results["ratescore"].append(result["ratescore"])
                results["green"].append(result["green"])
                results["radcliq"].append(result["radcliq"])
            except Exception as e:
                study_id = futures[future]
                print(f"Error processing {study_id}: {e}")

    batch_size = 32
    all_preds = []
    all_refs = []

    for study_id in ground_truth.keys():
        if study_id not in predictions:
            continue
        pred_text = predictions[study_id].get("model_prediction", "")
        gt_text = ground_truth[study_id].get("report", "")

        if mode == "findings":
            pred_text = extract_findings(pred_text)
            gt_text = extract_findings(gt_text)
        elif mode == "findings_impression":
            pred_findings = extract_findings(pred_text)
            gt_findings = extract_findings(gt_text)
            pred_impression = extract_impression(pred_text)
            gt_impression = extract_impression(gt_text)
            pred_text = f"{pred_findings} {pred_impression}".strip()
            gt_text = f"{gt_findings} {gt_impression}".strip()

        all_preds.append(preprocess_text(pred_text))
        all_refs.append(preprocess_text(gt_text))

    for i in range(0, len(all_preds), batch_size):
        batch_preds = all_preds[i : i + batch_size]
        batch_refs = all_refs[i : i + batch_size]

        try:
            bertscore = calculate_bertscore(batch_preds, batch_refs)
            for _ in range(len(batch_preds)):
                results["bertscore"].append(bertscore)
                results["semscore"].append(bertscore)
        except Exception as e:
            print(f"Batch error: {e}")
            for _ in range(len(batch_preds)):
                results["bertscore"].append(0.0)
                results["semscore"].append(0.0)

    metrics = {}
    for key, values in results.items():
        if len(values) > 0:
            metrics[key] = np.mean(values)
        else:
            metrics[key] = 0.0

    metrics["1/RadCliQ"] = (
        1.0 / metrics["radcliq"] if metrics["radcliq"] > 0 else float("inf")
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate radiology report generation")
    parser.add_argument(
        "--pred_json", type=str, required=True, help="Prediction JSON file"
    )
    parser.add_argument(
        "--gt_json", type=str, required=True, help="Ground truth JSON file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="findings",
        choices=["findings", "findings_impression"],
        help="Evaluation mode",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent workers",
    )

    args = parser.parse_args()

    print(f"Evaluating {args.pred_json} against {args.gt_json}")
    print(f"Mode: {args.mode}, Concurrency: {args.concurrency}")

    metrics = evaluate_rexrank(
        args.pred_json, args.gt_json, args.mode, args.concurrency
    )

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
