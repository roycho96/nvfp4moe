import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from real_model_benchmark import _result_slug
from real_model_capture import DATASETS, _dataset_text


def test_dataset_matrix_covers_practical_domains():
    assert set(DATASETS) == {
        "fineweb_edu",
        "c4_en",
        "finemath_4plus",
        "fineweb2_ko",
        "stack_v3_code",
    }


def test_plain_dataset_text():
    dataset = DATASETS["finemath_4plus"]
    assert _dataset_text(dataset, {"text": "x + y = 3"}) == "x + y = 3"


def test_stack_dataset_skips_vendor_files():
    dataset = DATASETS["stack_v3_code"]
    row = {
        "files": [
            {"content": "def train(): pass", "is_vendor": False},
            {"content": "minified dependency", "is_vendor": True},
            {"content": "class Router: pass", "is_vendor": False},
        ]
    }
    assert _dataset_text(dataset, row) == "def train(): pass\n\nclass Router: pass"


def test_result_slug_keeps_dataset_identity():
    assert _result_slug(["qwen3_30b_a3b"]) == "qwen3_30b_a3b__fineweb_edu"
    assert _result_slug(["qwen3_30b_a3b"], "c4_en") == "qwen3_30b_a3b__c4_en"
