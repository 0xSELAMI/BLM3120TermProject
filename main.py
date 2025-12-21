#!/usr/bin/env python

import sys
import argparse

from sklearn.model_selection import train_test_split

from common.Instance import Instance
from common.Features import FeatureType, FeatureFilter

from common.Dataset import DatasetSchema, Dataset

from common.Utils import save_dataset, load_dataset, process_dataset, create_test_and_train_set

from decision_tree.DecisionTree import build_decision_tree, evaluate_decision_tree

from CAR.CAR import generate_CARs, evaluate_CARs

default_trainset_path = "dataset/trainset.json"
default_testset_path = "dataset/testset.json"

def create_process_dataset_argparser(parsers, subparsers):
    default_testset_trainset_ratio = 0.2

    main_desc = "Create testset and trainset from supplied dataset"

    parsers["process_dataset"] = subparsers.add_parser("process_dataset", description=main_desc, help=main_desc)
    parsers["process_dataset"].add_argument("--dataset", "-d", metavar='PATH_TO_DATASET', type=str, required=True)
    parsers["process_dataset"].add_argument("--ratio", "-r", metavar='RATIO', help=f"ratio of the size of the testset to the size of whole dataset (default: {default_testset_trainset_ratio})", default=default_testset_trainset_ratio, type=float)
    parsers["process_dataset"].add_argument("--trainset-outfile", metavar='TRAINSET_OUTPATH', help=f"default: {default_trainset_path}", default=default_trainset_path, type=str)
    parsers["process_dataset"].add_argument("--testset-outfile", metavar='TESTSET_OUTPATH', help=f"default: {default_testset_path}", default=default_testset_path, type=str)

def create_decision_tree_argparser(parsers, subparsers):
    main_desc  = "Build or evaluate a decision tree"
    build_desc = "Build the decision tree and save into a pickle file, also create a DOT file"
    eval_desc  = "Evaluate the supplied decision tree using the test set"

    parsers["decision_tree"]                = {}
    parsers["decision_tree"]["main_parser"] = subparsers.add_parser("decision_tree", description=main_desc, help=main_desc)

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

    parsers["decision_tree"]["build_tree"] = decision_tree_subparsers.add_parser("build_tree", description=build_desc, help=build_desc)
    parsers["decision_tree"]["build_tree"].add_argument("--use-gini", action='store_true', help=f"Use Gini impurity instead of entropy (default: {default_use_gini})", default=default_use_gini)
    parsers["decision_tree"]["build_tree"].add_argument("--entropy-weights", "-e", nargs=2, metavar=('WEIGHT_TRUE', 'WEIGHT_FALSE'), help=f"Entropy weights for true and false labels respectively (default: {default_entropy_weights})", default=default_entropy_weights, type=float)
    parsers["decision_tree"]["build_tree"].add_argument("--max-depth", metavar='MAX_DEPTH', help=f"Max decision tree depth (default: {default_max_depth})", default=default_max_depth, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--min-info-gain", "-g", metavar='MIN_INFO_GAIN', help=f"Minimum info gain for a split to qualify as one (default: {default_min_info_gain})", default=default_min_info_gain, type=float)
    parsers["decision_tree"]["build_tree"].add_argument("--min-samples-split", metavar='MIN_SAMPLES_SPLIT', help=f"Minimum samples a meaningful split should have (default: {default_min_samples_split})", default=default_min_samples_split, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--min-samples-leaf", metavar='MIN_SAMPLES_LEAF', help=f"Minimum samples a leaf node should have (default: {default_min_samples_leaf})", default=default_min_samples_leaf, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--min-samples-leaf-kary", metavar='MIN_SAMPLES_LEAF_KARY', help=f"Minimum samples a leaf node of a k-ary node should have (default: {default_min_samples_leaf_kary})", default=default_min_samples_leaf_kary, type=int)
    parsers["decision_tree"]["build_tree"].add_argument("--dot-outfile", "-o", metavar='DOT_OUTPUT_FILEPATH', help=f"Path to write the dotfile of the decision tree to (default: {default_dot_outfile})", default=default_dot_outfile, type=str)
    parsers["decision_tree"]["build_tree"].add_argument("--pickle-path", "-p", metavar='PICKLE_FILEPATH', help=f"Path to pickle the decision tree to (default: {default_pickle_path})", default=default_pickle_path, type=str)
    parsers["decision_tree"]["build_tree"].add_argument("--trainset-infile", "-i", metavar='TRAINSET_FILEPATH', help=f"default: {default_trainset_path}", default=default_trainset_path, type=str)


    parsers["decision_tree"]["evaluate_tree"] = decision_tree_subparsers.add_parser("evaluate_tree", description=eval_desc, help=eval_desc)
    parsers["decision_tree"]["evaluate_tree"].add_argument("--pickle-path", "-p", metavar='PICKLE_FILEPATH', help=f"Path to pickled decision tree (default: {default_pickle_path})", default=default_pickle_path, type=str)
    parsers["decision_tree"]["evaluate_tree"].add_argument("--testset-infile", "-i", metavar='TESTSET_FILEPATH', help=f"default: {default_testset_path}", default=default_testset_path, type=str)

def create_CAR_argparser(parsers, subparsers):
    # default_use_gini              = False
    default_entropy_weights = [3.0, 1.0]
    default_max_split_count = 5
    default_min_bin_frac    = 0.1
    default_delta_cost      = 0.03

    default_max_k           = 3
    default_min_support     = 0.05
    default_min_confidence  = 0.8

    default_pickle_path = "CAR/rules.pickle"

    main_desc     = "Generate CARs (Class Association Rules) or evaluate CARs"
    generate_desc = "Generate CARs (Class Association Rules) and save into a pickle file"
    eval_desc     = "Evaluate the supplied CARs using the test set"

    parsers["CAR"]                = {}
    parsers["CAR"]["main_parser"] = subparsers.add_parser("CAR", description=main_desc, help=main_desc)

    CAR_subparsers = parsers["CAR"]["main_parser"].add_subparsers(title="commands", dest="subcommand_CAR")

    parsers["CAR"]["gen"] = CAR_subparsers.add_parser("generate", description=generate_desc, help=generate_desc)
    parsers["CAR"]["gen"].add_argument("--trainset-infile", metavar='TRAINSET_FILEPATH', help=f"default: {default_trainset_path}", default=default_trainset_path, type=str)
    parsers["CAR"]["gen"].add_argument("--pickle-path", metavar='PICKLE_PATH', help=f"default: {default_pickle_path}", default=default_pickle_path, type=str)
    parsers["CAR"]["gen"].add_argument("--entropy-weights", "-e", nargs=2, metavar=('WEIGHT_TRUE', 'WEIGHT_FALSE'), help=f"Entropy weights to use for true and false labels respectively while discretizing numeric features (default: {default_entropy_weights})", default=default_entropy_weights, type=float)
    parsers["CAR"]["gen"].add_argument("--max-split-count", "-m", metavar='MAX_SPLIT_COUNT', help=f"Max split count to consider while discretizing numeric features (default: {default_max_split_count})", default=default_max_split_count, type=int)
    parsers["CAR"]["gen"].add_argument("--min-bin-frac", metavar='MIN_BIN_FRACTION', help=f"Minimum fraction of the training dataset a bin should cover while discretizing numeric features into multiple bins (default: {default_min_bin_frac})", default=default_min_bin_frac, type=float)
    parsers["CAR"]["gen"].add_argument("--delta_cost", metavar='DELTA_COST', help=f"Minimum cost difference adding a new bin should make while discretizing numeric features into multiple bins (default: {default_delta_cost})", default=default_delta_cost, type=float)
    parsers["CAR"]["gen"].add_argument("--max-k", metavar='MAX_K', help=f"Max k value for the apriori algorithm (default: {default_max_k})", default=default_max_k, type=int)
    parsers["CAR"]["gen"].add_argument("--min-support", metavar='MIN_SUP', help=f"Minimum support for the CARs (default: {default_min_support})", default=default_min_support, type=float)
    parsers["CAR"]["gen"].add_argument("--min-confidence", metavar='MIN_CONF', help=f"Minimum confidence for the CARs (default: {default_min_confidence})", default=default_min_confidence, type=float)

    parsers["CAR"]["eval"] = CAR_subparsers.add_parser("evaluate", description=eval_desc, help=eval_desc)
    parsers["CAR"]["eval"].add_argument("--testset-infile", metavar='TESTSET_FILEPATH', help=f"default: {default_testset_path}", default=default_testset_path, type=str)

def create_naive_bayesian_argparser(parsers, subparsers):
    parsers["naive_bayesian"] = subparsers.add_parser("naive_bayesian", description="WIP", help="WIP")

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
    create_decision_tree_argparser(parsers, subparsers)
    create_CAR_argparser(parsers, subparsers)
    create_naive_bayesian_argparser(parsers, subparsers)

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

    elif args.command == "CAR":
        if args.subcommand_CAR == "generate":
            generate_CARs(args)
        elif args.subcommand_CAR == "evaluate":
            evaluate_CARs(args)
        else:
            parsers["CAR"]["main_parser"].print_help()

    elif args.command == "naive_bayesian":
        # TODO
        pass
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
