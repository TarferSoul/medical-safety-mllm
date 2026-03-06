"""GREEN metric via vLLM OpenAI-compatible API.

Constructs prompts matching the GREEN library format, sends them to a
vLLM-served GREEN-radllama2-7b model, and parses the green_score from
the LLM response.
"""

import re
from typing import List, Optional

import openai


def make_prompt(reference: str, prediction: str, max_len: int = 300) -> str:
    """Construct the GREEN evaluation prompt.

    Matches the prompt format from green_score/utils.py.
    """
    text1 = " ".join(reference.split()[:max_len])
    text2 = " ".join(prediction.split()[:max_len])
    prompt = (
        "Objective: Evaluate the accuracy of a candidate radiology report "
        "in comparison to a reference radiology report composed by expert radiologists.\n\n"
        "    Process Overview: You will be presented with:\n\n"
        "    1. The criteria for making a judgment.\n"
        "    2. The reference radiology report.\n"
        "    3. The candidate radiology report.\n"
        "    4. The desired format for your assessment.\n\n"
        "    1. Criteria for Judgment:\n\n"
        "    For each candidate report, determine:\n\n"
        "    The count of clinically significant errors.\n"
        "    The count of clinically insignificant errors.\n\n"
        "    Errors can fall into one of these categories:\n\n"
        "    a) False report of a finding in the candidate.\n"
        "    b) Missing a finding present in the reference.\n"
        "    c) Misidentification of a finding's anatomic location/position.\n"
        "    d) Misassessment of the severity of a finding.\n"
        "    e) Mentioning a comparison that isn't in the reference.\n"
        "    f) Omitting a comparison detailing a change from a prior study.\n"
        "    Note: Concentrate on the clinical findings rather than the report's "
        "writing style. Evaluate only the findings that appear in both reports.\n\n"
        f"    2. Reference Report:\n    {text1}\n\n"
        f"    3. Candidate Report:\n    {text2}\n\n"
        "    4. Reporting Your Assessment:\n\n"
        "    Follow this specific format for your output, even if no errors are found:\n"
        "    ```\n"
        "    [Explanation]:\n"
        "    <Explanation>\n\n"
        "    [Clinically Significant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n"
        "    [Clinically Insignificant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n"
        "    [Matched Findings]:\n"
        "    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n"
        "    ```\n"
    )
    return prompt


def clean_response(response: str) -> str:
    """Clean a GREEN model response (mirrors green_score/utils.py)."""
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def parse_green_score(response: str) -> Optional[float]:
    """Parse a green_score from a GREEN model response.

    GREEN_score = matched_findings / (matched_findings + sig_errors)
    """
    def _parse_section(text: str, category: str):
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return 0, [0] * 6
        content = match.group(1)
        if content.startswith("No"):
            return 0, [0] * 6
        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", content)
            return int(counts[0]) if counts else 0, [0] * 6
        sub_cats = [f"({c}) " for c in "abcdef"]
        matches = sorted(re.findall(r"\([a-f]\) .*", content))
        if not matches:
            matches = sorted(re.findall(r"\([1-6]\) .*", content))
            sub_cats = [f"({i}) " for i in range(1, 7)]
        sub_counts = [0] * 6
        for pos, sc in enumerate(sub_cats):
            for m in matches:
                if m.startswith(sc):
                    count = re.findall(r"(?<=: )\b\d+\b(?=\.)", m)
                    if count:
                        sub_counts[pos] = int(count[0])
        return sum(sub_counts), sub_counts

    _, sig_errors = _parse_section(response, "Clinically Significant Errors")
    matched, _ = _parse_section(response, "Matched Findings")

    if matched == 0:
        return 0.0
    return matched / (matched + sum(sig_errors))


class GREENMetric:
    """GREEN metric via vLLM HTTP API."""

    def __init__(self, base_url: str):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="unused",  # vLLM doesn't require a real key
        )
        # Detect the model name served by vLLM
        models = self.client.models.list()
        self.model_name = models.data[0].id if models.data else "default"

    def compute(self, predictions: List[str], references: List[str]) -> List[float]:
        """Compute per-sample GREEN scores via vLLM.

        Args:
            predictions: Generated reports.
            references: Ground-truth reports.

        Returns:
            List of GREEN scores (0-1).
        """
        scores = []
        for pred, ref in zip(predictions, references):
            prompt = make_prompt(ref, pred)
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=1024,
                    temperature=0,
                )
                text = response.choices[0].text
                text = clean_response(text)
                score = parse_green_score(text)
                scores.append(score if score is not None else 0.0)
            except Exception:
                scores.append(0.0)
        return scores
