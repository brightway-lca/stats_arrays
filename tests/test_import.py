import subprocess
import sys


def test_import_does_not_load_scipy_stats():
    # scipy.stats takes about half a second to import; only the cdf, ppf and pdf
    # methods need it. A fresh interpreter, since the test session may load it.
    code = "import sys, stats_arrays; assert 'scipy.stats' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
