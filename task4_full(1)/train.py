
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml

from methods.vanilla import train as train_vanilla
from methods.gcsc import train_gcsc
from methods.proser import train as train_proser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    method = cfg["method"]
    if method == "vanilla":
        train_vanilla(cfg)
    elif method == "gcsc":
        train_gcsc(cfg)
    elif method == "proser":
        train_proser(cfg)
    else:
        raise ValueError(f"unknown method: {method}")


if __name__ == "__main__":
    main()
