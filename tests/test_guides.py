import runpy
from pathlib import Path

import pytest

path_to_guides = Path(__file__).parent.parent / "guides"
scripts = path_to_guides.rglob("*.py")


@pytest.mark.parametrize("script", scripts)
def test_script_execution(script):
    print(f"Script: {script!s}")

    runpy.run_path(str(script))
