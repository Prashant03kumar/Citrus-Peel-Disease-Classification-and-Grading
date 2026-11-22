import json
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRADING_CONFIG_PATH = PROJECT_ROOT / "backend" / "grading_config.json"


class GradeDecision(TypedDict):
    grade: str
    decision: str


class GradingConfig(TypedDict):
    confidence_threshold: float
    rules: dict[str, GradeDecision]


def load_grading_config(path: Path = GRADING_CONFIG_PATH) -> GradingConfig:
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in grading configuration {path}.")
    threshold = config.get("confidence_threshold")
    rules = config.get("rules")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise ValueError(f"Invalid confidence_threshold in {path}; expected a number in [0, 1].")
    if not isinstance(rules, dict):
        raise ValueError(f"Invalid rules in {path}; expected a class-to-decision object.")
    validated_rules: dict[str, GradeDecision] = {}
    for class_name, rule in rules.items():
        if not isinstance(class_name, str) or not isinstance(rule, dict):
            raise ValueError(f"Invalid class rule in grading configuration {path}.")
        if not isinstance(rule.get("grade"), str) or not isinstance(
            rule.get("decision"), str
        ):
            raise ValueError(f"Each rule in {path} needs string grade and decision fields.")
        validated_rules[class_name] = {
            "grade": rule["grade"],
            "decision": rule["decision"],
        }
    return {
        "confidence_threshold": float(threshold),
        "rules": validated_rules,
    }


def grade_prediction(class_name: str, confidence: float) -> GradeDecision:
    config = load_grading_config()
    if confidence < float(config["confidence_threshold"]):
        return {"grade": "Review", "decision": "MANUAL INSPECTION"}

    rule = config["rules"].get(class_name)
    if rule is None:
        return {"grade": "Review", "decision": "MANUAL INSPECTION"}
    return rule
