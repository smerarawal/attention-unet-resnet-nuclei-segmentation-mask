"""python run_eda.py --data_root /kaggle/input/<monusac-dataset>"""
from nucseg.config import parse_cli
from nucseg.eda import run_eda

if __name__ == "__main__":
    run_eda(parse_cli())
