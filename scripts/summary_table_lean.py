import argparse
import sys
sys.path.append('.')
from ai4mat.common.summary_table import do_table, get_argparser

def main():
    args = get_argparser().parse_args()
    do_table(args)

if __name__ == "__main__":
    main()