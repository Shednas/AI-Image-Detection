# The four model files are duplicated in app/ and research/ so each tree runs alone.
# Comparing the parsed AST ignores comments and formatting, catching only the drift
# that would let a checkpoint load into a mismatched class.

import ast
from pathlib import Path

import pytest

MODEL_FILES = ["cnn_model.py", "fft_model.py", "hybrid_model.py", "stm_model.py"]

APP_MODELS = Path(__file__).resolve().parents[1] / "models"
RESEARCH_MODELS = Path(__file__).resolve().parents[3] / "research" / "src" / "models"


def structure(path: Path) -> str:
    return ast.dump(ast.parse(path.read_text(encoding="utf-8")), indent=2)


# Fails if a model class was edited in one tree only, which is how a checkpoint ends up
# loading into the wrong architecture.
@pytest.mark.parametrize("filename", MODEL_FILES)
def test_app_and_research_model_definitions_match(filename):
    if not RESEARCH_MODELS.is_dir():
        pytest.skip("research/ is not present in this checkout")

    app_file = APP_MODELS / filename
    research_file = RESEARCH_MODELS / filename
    assert app_file.is_file(), f"{app_file} is missing"
    assert research_file.is_file(), f"{research_file} is missing"

    assert structure(app_file) == structure(research_file), (
        f"{filename} has diverged between app/ and research/. "
        "Sync the two copies rather than editing one."
    )
