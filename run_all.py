"""EDA -> train -> evaluate(test). One command."""
from nucseg.config import parse_cli
from nucseg.eda import run_eda
from nucseg.engine import evaluate, prepare, train

if __name__ == "__main__":
    cfg = parse_cli()
    data = prepare(cfg)
    run_eda(cfg, data)
    train(cfg, data)
    evaluate(cfg, "val", data=data)
    if "test" in data:
        evaluate(cfg, "test", data=data)
