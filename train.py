"""python train.py --epochs 60 --encoder resnet34   (add --resume true to continue)"""
from nucseg.config import parse_cli
from nucseg.engine import train

if __name__ == "__main__":
    train(parse_cli())
