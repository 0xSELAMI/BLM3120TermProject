#!/usr/bin/env python

import re
import sys
import argparse

from sklearn.model_selection import train_test_split

from Instance import Instance
from Features import FeatureType, FeatureFilter

from Dataset import DatasetSchema, Dataset

from Utils import save_dataset, load_dataset, process_dataset, create_test_and_train_set

# from DecisionTree import decision_tree

def main():
    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = """
                            A python script to oversee and fulfill the functionalities
                            the project proposal document specifies
                        """,
                        epilog='StudentID: [REDACTED]')

    subparsers = parser.add_subparsers(title="commands", dest="command")

    parser_process_dataset = subparsers.add_parser("process_dataset", description="Create testset and trainset from supplied dataset", help="Create testset and trainset from supplied dataset")
    parser_process_dataset.add_argument("--dataset", "-d", metavar='PATH_TO_DATASET', type=str, required=True)
    parser_process_dataset.add_argument("--ratio", "-r", metavar='RATIO', help="ratio of the size of the testset to the size of whole dataset (default: 0.2)", default=0.2, type=float)
    parser_process_dataset.add_argument("--trainset-outfile", metavar='TRAINSET_OUTPATH', help="default: dataset/trainset.json", default="dataset/trainset.json", type=str)
    parser_process_dataset.add_argument("--testset-outfile", metavar='TESTSET_OUTPATH', help="default: dataset/testset.json", default="dataset/testset.json", type=str)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--testset-infile", metavar='TESTSET_FILEPATH', help="default: dataset/testset.json", default="dataset/testset.json", type=str)
    common_parser.add_argument("--trainset-infile", metavar='TRAINSET_FILEPATH', help="default: dataset/trainset.json", default="dataset/trainset.json", type=str)

    parser_process_dataset = subparsers.add_parser("decision_tree", description="WIP", help="WIP", parents=[common_parser])

    args = parser.parse_args()

    if args.command == "process_dataset":
        process_dataset(args)
    elif args.command == "decision_tree":
        # decision_tree(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
