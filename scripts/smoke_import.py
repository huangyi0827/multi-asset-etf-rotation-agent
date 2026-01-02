import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from vendor.GeneralBacktest.backtest import GeneralBacktest

print("OK: imported GeneralBacktest:", GeneralBacktest)
