#!/usr/bin/env python

import re
import sys
import argparse

from sklearn.model_selection import train_test_split

from common.Instance import Instance
from common.Features import FeatureType, FeatureFilter

from common.Dataset import DatasetSchema, Dataset

from common.Utils import save_dataset, load_dataset, process_dataset, create_test_and_train_set

from decision_tree.DecisionTree import build_decision_tree, evaluate_decision_tree

def create_process_dataset_argparser(parsers, subparsers):
    default_testset_trainset_ratio = 0.2

    parsers["process_dataset"] = subparsers.add_parser("process_dataset", description="Create testset and trainset from supplied dataset", help="Create testset and trainset from supplied dataset")
    parsers["process_dataset"].add_argument("--dataset", "-d", metavar='PATH_TO_DATASET', type=str, required=True)
    parsers["process_dataset"].add_argument("--ratio", "-r", metavar='RATIO', help=f"ratio of the size of the testset to the size of whole dataset (default: {default_testset_trainset_ratio})", default=default_testset_trainset_ratio, type=float)
    parsers["process_dataset"].add_argument("--trainset-outfile", metavar='TRAINSET_OUTPATH', help="default: dataset/trainset.json", default="dataset/trainset.json", type=str)
    parsers["process_dataset"].add_argument("--testset-outfile", metavar='TESTSET_OUTPATH', help="default: dataset/testset.json", default="dataset/testset.json", type=str)

def create_decision_tree_argparser(parsers, subparsers):
    parsers["decision_tree"] = {}
    parsers["decision_tree"]["main_parser"] = subparsers.add_parser("decision_tree", description="Build or evaluate a decision tree", help="Build or evaluate a decision tree")

    decision_tree_subparsers = parsers["decision_tree"]["main_parser"].add_subparsers(title="commands", dest="subcommand_decision_tree")

    default_max_depth             = 24
    default_min_samples_split     = 4
    default_min_info_gain         = 1e-4
    default_min_samples_leaf      = 2
    default_min_samples_leaf_kary = 0

    default_use_gini              = False
    default_entropy_weights       = [3.0, 1.0]

    default_dot_outfile = "dotfiles/decisiontree.dot"

    default_pickle_path = "decision_tree/decisiontree.pickle"

    parsers["decision_tree"]["build_tree"] = decision_tree_subparsers.add_parser("build_tree", description="Build the decision tree and save into a pickle file, also create a DOT file", help="Build the decision tree and save into a pickle file, also create a DOT file")
    parsers["decision_tree"]["build_tree"].add_argument("--use-gini", action='store_true', help=f"Use Gini impurity instead of entropy (default: {default_use_gini})", default=default_use_gini)
    parsers["decision_tree"]["build_tree"].add_argument("--entropy-weights", "-e", nargs=2, metavar=('WEIGHT_TRUE', 'WEIGHT_FALSE'), help=f"Entropy weights for true and false labels respectively (default: {default_entropy_weights})", default=default_entropy_weights, type=float)
    parsers["decision_tree"]["build_tree"].add_argument("--max-depth", metavar='MAX_DEPTH', help=f"Max decision tree depth (default: {default_max_depth})", default=default_max_depth, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--min-info-gain", "-g", metavar='MIN_INFO_GAIN', help=f"Minimum info gain for a split to qualify as one (default: {default_min_info_gain})", default=default_min_info_gain, type=float)
    parsers["decision_tree"]["build_tree"].add_argument("--min-samples-split", metavar='MIN_SAMPLES_SPLIT', help=f"Minimum samples a meaningful split should have (default: {default_min_samples_split})", default=default_min_samples_split, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--min-samples-leaf", metavar='MIN_SAMPLES_LEAF', help=f"Minimum samples a leaf node should have (default: {default_min_samples_leaf})", default=default_min_samples_leaf, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--min-samples-leaf-kary", metavar='MIN_SAMPLES_LEAF_KARY', help=f"Minimum samples a leaf node of a k-ary node should have (default: {default_min_samples_leaf_kary})", default=default_min_samples_leaf_kary, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--dot-outfile", "-o", metavar='DOT_OUTPUT_FILEPATH', help=f"Path to write the dotfile of the decision tree to (default: {default_dot_outfile})", default=default_dot_outfile, type=str)
    parsers["decision_tree"]["build_tree"].add_argument("--pickle-path", "-p", metavar='PICKLE_FILEPATH', help=f"Path to pickle the decision tree to (default: {default_pickle_path})", default=default_pickle_path, type=str)
    parsers["decision_tree"]["build_tree"].add_argument("--trainset-infile", "-i", metavar='TRAINSET_FILEPATH', help="default: dataset/trainset.json", default="dataset/trainset.json", type=str)

    parsers["decision_tree"]["evaluate_tree"] = decision_tree_subparsers.add_parser("evaluate_tree", description="Evaluate the supplied decision tree using the test set", help="Evaluate the supplied decision tree using the test set")
    parsers["decision_tree"]["evaluate_tree"].add_argument("--pickle-path", "-p", metavar='PICKLE_FILEPATH', help=f"Path to pickled decision tree (default: {default_pickle_path})", default=default_pickle_path, type=str)
    parsers["decision_tree"]["evaluate_tree"].add_argument("--testset-infile", "-i", metavar='TESTSET_FILEPATH', help="default: dataset/testset.json", default="dataset/testset.json", type=str)


def main():
    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = """
                            A python script to oversee and fulfill the functionalities
                            the project proposal document specifies
                        """,
                        epilog='StudentID: [REDACTED]')

    subparsers = parser.add_subparsers(title="commands", dest="command")

    parsers = {}

    create_process_dataset_argparser(parsers, subparsers)

    # common_parser = argparse.ArgumentParser(add_help=False)
    # common_parser.add_argument("--testset-infile", metavar='TESTSET_FILEPATH', help="default: dataset/testset.json", default="dataset/testset.json", type=str)
    # common_parser.add_argument("--trainset-infile", metavar='TRAINSET_FILEPATH', help="default: dataset/trainset.json", default="dataset/trainset.json", type=str)

    create_decision_tree_argparser(parsers, subparsers)

    args = parser.parse_args()

    if args.command == "process_dataset":
        process_dataset(args)
    elif args.command == "decision_tree":
        if args.subcommand_decision_tree == "build_tree":
            build_decision_tree(args)
        elif args.subcommand_decision_tree == "evaluate_tree":
            evaluate_decision_tree(args)
        else:
            parsers["decision_tree"]["main_parser"].print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
