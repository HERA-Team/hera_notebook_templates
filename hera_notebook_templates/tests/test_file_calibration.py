from subprocess import run
from pathlib import Path
import os

DATA_PATH = Path(__file__).parent / 'data'
NOTEBOOKS = Path(__file__).parent.parent.parent / 'notebooks'

def test_that_filecal_runs_through(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "zen.2459847.51336.sum.html"
    result = run(
    [
        "jupyter", "nbconvert",
        "--to", "html",
        "--execute",
        "--output", str(fn),
        str(NOTEBOOKS/'file_calibration.ipynb')
    ],
    env={
        **os.environ,
        **{
            "SUM_FILE": str(DATA_PATH/'zen.2459847.51336.sum.downsampled.uvh5'),
            "AUTO_SLOPE_GOOD_LOW": "-1.0",  # Set these wider than usual, because we're
            "AUTO_SLOPE_GOOD_HIGH": "1.0",  # using a subset of frequency.
            "OC_MIN_DIM_SIZE": "0",         # Default is 8, but we don't have that many redundant bls in our small set.
        }
    }
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0