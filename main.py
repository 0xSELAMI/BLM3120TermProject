#!/usr/bin/env python

import sys
import argparse

from sklearn.model_selection import train_test_split

from common.Instance import Instance
from common.Features import FeatureType, FeatureFilter

from common.Dataset import DatasetSchema, Dataset

from common.Utils import save_dataset, load_dataset, process_dataset, create_test_and_train_set

from decision_tree.DecisionTree import build_decision_tree, evaluate_decision_tree

from CBA.CBA import generate_CARs, evaluate_CARs

from naive_bayesian.NaiveBayesian import build_naive_bayesian_classifier, evaluate_naive_bayesian_classifier

def create_process_dataset_argparser(parsers, subparsers, default_trainset_path, default_testset_path):
    default_testset_trainset_ratio = 0.2

    main_desc = "Create testset and trainset from supplied dataset"

    parsers["process_dataset"] = subparsers.add_parser("process_dataset", description=main_desc, help=main_desc)
    parsers["process_dataset"].add_argument("--dataset", "-d", metavar='PATH_TO_DATASET', type=str, required=True)
    parsers["process_dataset"].add_argument("--ratio", "-r", metavar='RATIO', help=f"ratio of the size of the testset to the size of whole dataset (default: {default_testset_trainset_ratio})", default=default_testset_trainset_ratio, type=float)
    parsers["process_dataset"].add_argument("--trainset-outfile", metavar='TRAINSET_OUTPATH', help=f"default: {default_trainset_path}", default=default_trainset_path, type=str)
    parsers["process_dataset"].add_argument("--testset-outfile", metavar='TESTSET_OUTPATH', help=f"default: {default_testset_path}", default=default_testset_path, type=str)

def create_decision_tree_argparser(parsers, subparsers, parent_parsers):
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

    pickle_parser = argparse.ArgumentParser(add_help=False)
    pickle_parser.add_argument("--pickle-path", metavar='PICKLE_PATH', help=f"default: {default_pickle_path}", default=default_pickle_path, type=str)

    parsers["decision_tree"]["build"] = decision_tree_subparsers.add_parser("build", description=build_desc, help=build_desc, parents=[parent_parsers["builder"], pickle_parser])
    parsers["decision_tree"]["build"].add_argument("--use-gini", action='store_true', help=f"Use Gini impurity instead of entropy (default: {default_use_gini})", default=default_use_gini)
    parsers["decision_tree"]["build"].add_argument("--entropy-weights", "-e", nargs=2, metavar=('WEIGHT_TRUE', 'WEIGHT_FALSE'), help=f"Entropy weights for true and false labels respectively (default: {default_entropy_weights})", default=default_entropy_weights, type=float)
    parsers["decision_tree"]["build"].add_argument("--max-depth", metavar='MAX_DEPTH', help=f"Max decision tree depth (default: {default_max_depth})", default=default_max_depth, type=int)
    parsers["decision_tree"]["build"].add_argument("--min-info-gain", "-g", metavar='MIN_INFO_GAIN', help=f"Minimum info gain for a split to qualify as one (default: {default_min_info_gain})", default=default_min_info_gain, type=float)
    parsers["decision_tree"]["build"].add_argument("--min-samples-split", metavar='MIN_SAMPLES_SPLIT', help=f"Minimum samples a meaningful split should have (default: {default_min_samples_split})", default=default_min_samples_split, type=int)
    parsers["decision_tree"]["build"].add_argument("--min-samples-leaf", metavar='MIN_SAMPLES_LEAF', help=f"Minimum samples a leaf node should have (default: {default_min_samples_leaf})", default=default_min_samples_leaf, type=int)
    parsers["decision_tree"]["build"].add_argument("--min-samples-leaf-kary", metavar='MIN_SAMPLES_LEAF_KARY', help=f"Minimum samples a leaf node of a k-ary node should have (default: {default_min_samples_leaf_kary})", default=default_min_samples_leaf_kary, type=int)
    parsers["decision_tree"]["build"].add_argument("--dot-outfile", "-o", metavar='DOT_OUTPUT_FILEPATH', help=f"Path to write the dotfile of the decision tree to (default: {default_dot_outfile})", default=default_dot_outfile, type=str)


    parsers["decision_tree"]["evaluate"] = decision_tree_subparsers.add_parser("evaluate", description=eval_desc, help=eval_desc, parents=[parent_parsers["evaluator"], pickle_parser])

def create_CBA_argparser(parsers, subparsers, parent_parsers):
    default_max_k           = 4
    default_min_support     = 1e-3
    default_min_confidence  = 0.27
    default_min_lift        = 1.20
    default_error_weights   = [1.0, 1.5] # a false negative is %50 worse than a false positive

    default_pickle_path = "CBA/rules.pickle"

    main_desc     = "Generate a CAR (Class Association Rule) classifier or evaluate a CAR classifier"
    generate_desc = "Generate a classifier and save into a pickle file"
    eval_desc     = "Evaluate the supplied classifier using the test set"

    pickle_parser = argparse.ArgumentParser(add_help=False)
    pickle_parser.add_argument("--pickle-path", metavar='PICKLE_PATH', help=f"default: {default_pickle_path}", default=default_pickle_path, type=str)

    parsers["CBA"]                = {}
    parsers["CBA"]["main_parser"] = subparsers.add_parser("CBA", description=main_desc, help=main_desc)

    CBA_subparsers = parsers["CBA"]["main_parser"].add_subparsers(title="commands", dest="subcommand_CBA")

    parsers["CBA"]["gen"] = CBA_subparsers.add_parser("generate", description=generate_desc, help=generate_desc, parents=[parent_parsers["builder"], parent_parsers["discretizer"], pickle_parser])
    parsers["CBA"]["gen"].add_argument("--max-k", metavar='MAX_K', help=f"Max k value for the apriori algorithm (default: {default_max_k})", default=default_max_k, type=int)
    parsers["CBA"]["gen"].add_argument("--min-support", metavar='MIN_SUP', help=f"Minimum support for the CARs (default: {default_min_support})", default=default_min_support, type=float)
    parsers["CBA"]["gen"].add_argument("--min-confidence", metavar='MIN_CONF', help=f"Minimum confidence for the CARs (default: {default_min_confidence})", default=default_min_confidence, type=float)
    parsers["CBA"]["gen"].add_argument("--min-lift", metavar='MIN_LIFT', help=f"Minimum lift for the CARs (default: {default_min_lift})", default=default_min_lift, type=float)
    parsers["CBA"]["gen"].add_argument("--error-weights", nargs=2, metavar=('WEIGHT_FALSE_POSITIVES', 'WEIGHT_FALSE_NEGATIVES'), help=f"The weights to use for penalizing rules that incorrectly cover instances while building CAR classifier (default: {default_error_weights})", default=default_error_weights, type=float)

    parsers["CBA"]["eval"] = CBA_subparsers.add_parser("evaluate", description=eval_desc, help=eval_desc, parents=[parent_parsers["evaluator"], pickle_parser])

def create_naive_bayesian_argparser(parsers, subparsers, parent_parsers):
    main_desc  = "Build a naive bayesian classifier probability table or evaluate one"
    build_desc = "Build a naive bayesian classifier probability table using the trainset and save into a pickle file"
    eval_desc  = "Evaluate a naive bayesian classifier probability table using the supplied testset"

    default_pickle_path = "naive_bayesian/probability_table.pickle"

    pickle_parser = argparse.ArgumentParser(add_help=False)
    pickle_parser.add_argument("--pickle-path", metavar='PICKLE_PATH', help=f"default: {default_pickle_path}", default=default_pickle_path, type=str)

    parsers["naive_bayesian"]                = {}
    parsers["naive_bayesian"]["main_parser"] = subparsers.add_parser("naive_bayesian", description=main_desc, help=main_desc)

    NB_subparsers = parsers["naive_bayesian"]["main_parser"].add_subparsers(title="commands", dest="subcommand_NB")

    parsers["naive_bayesian"]["build"] = NB_subparsers.add_parser("build", description=build_desc, help=build_desc, parents=[parent_parsers["builder"], pickle_parser, parent_parsers["discretizer"]])

    parsers["naive_bayesian"]["evaluate"] = NB_subparsers.add_parser("evaluate", description=eval_desc, help=eval_desc, parents=[parent_parsers["evaluator"], pickle_parser])


def main():
    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = """
                            A python script to oversee and fulfill the functionalities
                            the project proposal document specifies
                        """,
                        epilog='StudentID: [REDACTED]')

    subparsers = parser.add_subparsers(title="commands", dest="command")

    default_trainset_path = "dataset/trainset.json"
    default_testset_path = "dataset/testset.json"

    default_entropy_weights = [3.0, 1.0]
    default_max_split_count = 3
    default_min_bin_frac    = 0.1
    default_delta_cost      = 1e-3

    parent_parsers = {}
    parent_parsers["builder"] = argparse.ArgumentParser(add_help=False)
    parent_parsers["builder"].add_argument("--trainset-infile", metavar='TRAINSET_FILEPATH', help=f"default: {default_trainset_path}", default=default_trainset_path, type=str)

    parent_parsers["evaluator"] = argparse.ArgumentParser(add_help=False)
    parent_parsers["evaluator"].add_argument("--testset-infile", metavar='TESTSET_FILEPATH', help=f"default: {default_testset_path}", default=default_testset_path, type=str)

    parent_parsers["discretizer"] = argparse.ArgumentParser(add_help=False)
    parent_parsers["discretizer"].add_argument("--entropy-weights", nargs=2, metavar=('WEIGHT_TRUE', 'WEIGHT_FALSE'), help=f"Entropy weights to use for true and false labels respectively while discretizing numeric features (default: {default_entropy_weights})", default=default_entropy_weights, type=float)
    parent_parsers["discretizer"].add_argument("--max-split-count", "-m", metavar='MAX_SPLIT_COUNT', help=f"Max split count to consider while discretizing numeric features (default: {default_max_split_count})", default=default_max_split_count, type=int)
    parent_parsers["discretizer"].add_argument("--min-bin-frac", metavar='MIN_BIN_FRACTION', help=f"Minimum fraction of the training dataset a bin should cover while discretizing numeric features into multiple bins (default: {default_min_bin_frac})", default=default_min_bin_frac, type=float)
    parent_parsers["discretizer"].add_argument("--delta-cost", metavar='DELTA_COST', help=f"Minimum cost difference adding a new bin should make while discretizing numeric features into multiple bins (default: {default_delta_cost})", default=default_delta_cost, type=float)

    parsers = {}

    create_process_dataset_argparser(parsers, subparsers, default_trainset_path, default_testset_path)
    create_decision_tree_argparser(parsers, subparsers, parent_parsers)
    create_CBA_argparser(parsers, subparsers, parent_parsers)
    create_naive_bayesian_argparser(parsers, subparsers, parent_parsers)

    args = parser.parse_args()

    if args.command == "process_dataset":
        process_dataset(args)
    elif args.command == "decision_tree":
        if args.subcommand_decision_tree == "build":
            build_decision_tree(args)
        elif args.subcommand_decision_tree == "evaluate":
            evaluate_decision_tree(args)
        else:
            parsers["decision_tree"]["main_parser"].print_help()

    elif args.command == "CBA":
        if args.subcommand_CBA == "generate":
            generate_CARs(args)
        elif args.subcommand_CBA == "evaluate":
            evaluate_CARs(args)
        else:
            parsers["CBA"]["main_parser"].print_help()

    elif args.command == "naive_bayesian":
        if args.subcommand_NB == "build":
            build_naive_bayesian_classifier(args)
        elif args.subcommand_NB == "evaluate":
            evaluate_naive_bayesian_classifier(args)
        else:
            parsers["naive_bayesian"]["main_parser"].print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
