import argparse
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Appends E_1 column to csf/cif dataset')
    parser.add_argument("--E1", type=Path, required=True,
                        help="Path to csv file with E_1 values")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset name")
    args = parser.parse_args()
    E_1 = pd.read_csv(args.E1, index_col=0).squeeze("columns")
