"""Run a community-detection experiment.

Example:
    python main.py --method secomm --dataset citeseer
"""

import argparse

from CD_models.NOCD_train import nocd
from CD_models.SEComm_train import secomm
from CD_models.VGAER_train import vgaer
from data_loader import load_dataset


METHODS = {"nocd": nocd, "secomm": secomm, "vgaer": vgaer}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a graph community-detection model on a supported dataset."
    )
    parser.add_argument("--method", choices=METHODS, default="secomm")
    parser.add_argument("--dataset", default="citeseer")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset)
    METHODS[args.method](dataset, args.dataset)


if __name__ == "__main__":
    main()
