"""python evaluate.py --split test   -> metrics json/csv + figures in <out_dir>/figures"""
import sys
from nucseg.config import parse_cli
from nucseg.engine import evaluate

if __name__ == "__main__":
    split = "test"
    if "--split" in sys.argv:
        split = sys.argv[sys.argv.index("--split") + 1]
    evaluate(parse_cli(), split=split)
