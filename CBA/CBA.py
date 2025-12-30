#from common.Utils import load_dataset, move_cursor_up_and_clear_line, save_pickle, load_pickle
import math
import common.Utils as CommonUtils

import common.Helpers as CommonHelpers
import common.Discretizer as Discretizer

from common.Transaction import apply_thresholds

import CBA.CBAHelpers as CBAHelpers

def generate_CARs(args):
    trainset  = CommonUtils.load_dataset(args.trainset_infile, args.entropy_weights)

    if not trainset:
        return None

    try:
        threshold_map         = Discretizer.best_thresholds_for_features(trainset, args.max_split_count, args.min_bin_frac, args.delta_cost)
        transactions          = apply_thresholds(trainset, threshold_map)
        
        max_k          = args.max_k
        min_support    = args.min_support
        min_confidence = args.min_confidence
        min_lift       = args.min_lift
        error_weights  = args.error_weights

        print(f"Running apriori algorithm... (max_k: {max_k}, min_support: {min_support}, min_confidence: {min_confidence}, min_lift: {min_lift})")
        # apriori returns all Fk where k in range (0, max_k)
        all_frequent_itemsets = CBAHelpers.apriori(transactions, min_support, max_k)
        CommonUtils.move_cursor_up_and_clear_line(2)
        print(f"Collected frequent itemsets up to size {len(all_frequent_itemsets)}. (max_k: {max_k}, min_support: {min_support}, min_confidence: {min_confidence}), min_lift: {min_lift}")

        all_rules, label_distribution = CBAHelpers.generate_rules(all_frequent_itemsets, transactions, min_support, min_confidence, min_lift)

        all_rules.sort(key = lambda r : (
            -r["confidence"],     # Accuracy first
            -r["support"],        # General trends over hyper-specific flukes
            -r["lift"],           # Strength of association
            -(r["label"] == True),# Prioritize finding subscribers
            len(r["itemset"]),    # Simple rules over complex ones
            str(r["itemset"])     # Deterministic tie-break
        ))

        rules, default_rule = CBAHelpers.build_classifier(all_rules, transactions, error_weights)
        rules.append( default_rule )

        print(f"Generated {len(all_rules)} rules. Down to {len(rules)} after building the classifier.\n")

        out = {"rules": rules, "threshold_map": threshold_map, "trainset_label_ratios": label_distribution}

        CommonUtils.save_pickle(out, args.pickle_path, "class association rules and treshold map")

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
        return None

def predict(transaction, rules):
    for rule in rules:
        # if rule has default: True, then return its label
        if rule.get("default", False):
            return rule["label"]

        # its not a default rule
        if rule["itemset"].issubset(transaction["itemset"]):
            return rule["label"]

# calculate the probability that a given transaction's label is true based on the rules
def predict_prob_transaction(rules, transaction, label_ratios):
    # 1. Filter for matching rules (excluding default)
    matching = [r for r in rules if not r.get("default") and r["itemset"].issubset(transaction["itemset"])]

    if not matching:
        return label_ratios[True]

    best_rule = matching[0]

    if best_rule["label"] == True:
        return best_rule["confidence"]
    else:
        # if the best rule predicts false with 0.8 conf, probability of true is 0.2
        return 1.0 - best_rule["confidence"]

def evaluate_CARs(args):
    try:
        testset        = CommonUtils.load_dataset(args.testset_infile)

        if not testset:
            return None

        pickled_data   = CommonUtils.load_pickle(args.pickle_path)

        if not pickled_data:
            return None

        rules                 = pickled_data["rules"]
        threshold_map         = pickled_data["threshold_map"]
        trainset_label_ratios = pickled_data["trainset_label_ratios"]

        transactions   = apply_thresholds(testset, threshold_map)

        predictions =   CommonHelpers.predict_dataset(
                            transactions, None, 
                            rules, predict
                        )


        accuracy, precision, recall = CommonHelpers.get_basic_metrics([t["label"] for t in transactions], predictions)

        values_and_probs =  CommonHelpers.get_label_values_and_probs (
                                rules, transactions,
                                None, lambda transaction: transaction["label"],
                                predict_prob_transaction, trainset_label_ratios
                            )

        roc_auc = CommonHelpers.calc_roc_auc(*values_and_probs)
        print(f"ROC-AUC: {round(roc_auc, 4)}")

        # TODO visualization, i think.
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
        exit(0)
