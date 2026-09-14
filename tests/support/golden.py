"""Golden-fixture helper for characterization tests (Plan.md Step 1).

The first run of a characterization test has nothing to compare against, so
it captures the current (legacy) output into a checked-in .pt file under
tests/characterization/golden/ and fails on purpose (RED) asking for a
rerun. The second run compares against that now-committed golden file and
passes (GREEN). From then on, any future code change that alters the
numeric output will make the test fail again.
"""

import json
import pathlib
import torch

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent.parent / "characterization" / "golden"


def assert_matches_golden(name: str, value: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-5) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.pt"
    value = value.detach().clone()
    if not path.exists():
        torch.save(value, path)
        raise AssertionError(
            f"golden fixture {path} did not exist and was just generated from the "
            f"current (legacy) code. Rerun the test to verify it now passes."
        )
    expected = torch.load(path, weights_only=True)
    assert torch.allclose(value, expected, atol=atol, rtol=rtol), (
        f"value no longer matches golden fixture {path}\n"
        f"value={value}\nexpected={expected}"
    )


def assert_matches_golden_json(name: str, value) -> None:
    """Like assert_matches_golden, but for JSON-serializable structural facts
    (e.g. [[param_name, shape], ...]) instead of numeric tensors."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False))
        raise AssertionError(
            f"golden fixture {path} did not exist and was just generated from the "
            f"current (legacy) code. Rerun the test to verify it now passes."
        )
    expected = json.loads(path.read_text())
    assert value == expected, (
        f"value no longer matches golden fixture {path}\n"
        f"value={value}\nexpected={expected}"
    )
