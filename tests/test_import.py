import os
import subprocess
import sys


def test_import_does_not_load_scipy():
    # scipy.stats takes about half a second to import; only the cdf, ppf and pdf
    # methods need it. A fresh interpreter, since the test session may load it.
    # Pass on our sys.path so the child imports the source tree under test, not
    # whatever copy of stats_arrays is installed.
    code = (
        "import sys, stats_arrays; "
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'scipy'); "
        "assert not loaded, f'{len(loaded)} scipy modules loaded: {loaded[:5]}'"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
