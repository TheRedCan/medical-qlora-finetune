"""GPU-free unit tests for the disease-extraction task."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from src.extraction import (  # noqa: E402
    bio_to_entities,
    build_extraction_prompt,
    build_target,
    normalize_ner_example,
    parse_diseases,
    score_example,
)
from src.stats import bootstrap_f1_diff, micro_f1  # noqa: E402


# ---- BIO -> entities --------------------------------------------------

def test_bio_to_entities_basic():
    toks = ["the", "postural", "hypotension", "and", "Parkinson", "disease", "."]
    tags = [0, 1, 2, 0, 1, 2, 0]
    assert bio_to_entities(toks, tags) == ["postural hypotension", "Parkinson disease"]


def test_bio_two_adjacent_entities_separated_by_B():
    toks = ["fever", "cough"]
    tags = [1, 1]  # two single-token entities back to back
    assert bio_to_entities(toks, tags) == ["fever", "cough"]


def test_bio_trailing_entity_closed():
    assert bio_to_entities(["x", "cancer"], [0, 1]) == ["cancer"]


def test_normalize_ner_dedup_preserves_order():
    ex = {"tokens": ["flu", "and", "flu", "again"], "tags": [1, 0, 1, 0]}
    norm = normalize_ner_example(ex)
    assert norm["text"] == "flu and flu again"
    assert norm["diseases"] == ["flu"]  # deduped


# ---- prompt / target --------------------------------------------------

def test_build_target_is_valid_json():
    import json
    t = build_target(["asthma", "COPD"])
    assert json.loads(t) == {"diseases": ["asthma", "COPD"]}


def test_extraction_prompt_ends_with_json_marker_then_target():
    ex = {"text": "patient has asthma", "diseases": ["asthma"]}
    p = build_extraction_prompt(ex, include_answer=False)
    assert p.rstrip().endswith("JSON:")
    full = build_extraction_prompt(ex, include_answer=True)
    assert full.startswith(p)
    assert full.strip().endswith('{"diseases": ["asthma"]}')


# ---- parsing model output --------------------------------------------

@pytest.mark.parametrize("gen,expected,valid", [
    ('{"diseases": ["asthma", "COPD"]}', ["asthma", "COPD"], True),
    ('Sure! {"diseases": ["flu"]} hope that helps', ["flu"], True),  # junk around JSON
    ('{"diseases": []}', [], True),
    ('the patient clearly has asthma', [], False),                    # no JSON
    ('{"wrong_key": ["x"]}', [], False),                              # wrong schema
    ('{"diseases": "asthma"}', [], False),                            # not a list
    ('', [], False),
])
def test_parse_diseases(gen, expected, valid):
    diseases, is_valid = parse_diseases(gen)
    assert diseases == expected
    assert is_valid == valid


# ---- scoring ----------------------------------------------------------

def test_score_example_case_insensitive_sets():
    sc = score_example(pred=["Asthma", "flu"], gold=["asthma", "COPD"])
    assert sc["tp"] == 1 and sc["fp"] == 1 and sc["fn"] == 1
    assert sc["exact"] is False


def test_score_example_exact_match():
    sc = score_example(pred=["asthma"], gold=["Asthma"])
    assert sc["exact"] is True and sc["tp"] == 1 and sc["fp"] == 0 and sc["fn"] == 0


# ---- micro F1 + bootstrap --------------------------------------------

def test_micro_f1_perfect_and_zero():
    assert micro_f1([(5, 0, 0)]) == 1.0
    assert micro_f1([(0, 3, 2)]) == 0.0


def test_bootstrap_detects_clear_f1_gain():
    # base extracts nothing (all FN); ft extracts everything right -> huge gain
    base = [(0, 0, 2)] * 200
    ft = [(2, 0, 0)] * 200
    r = bootstrap_f1_diff(base, ft, n_boot=500, seed=1)
    assert r["f1_base"] == 0.0 and r["f1_ft"] == 1.0
    assert r["f1_delta"] > 0.9
    assert r["significant_05"] is True


def test_bootstrap_no_gain_not_significant():
    same = [(1, 1, 1)] * 100
    r = bootstrap_f1_diff(same, same, n_boot=500, seed=1)
    assert r["f1_delta"] == 0.0
    assert r["significant_05"] is False
