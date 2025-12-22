import common.Utils as CommonUtils
import common.Helpers as CommonHelpers
import common.Discretizer as Discretizer
from common.Transaction import apply_thresholds

import math

def predict(transaction, probability_table, label_counts):
    scores = {label: math.log(0.5) for label in [True, False]}

    for label in [True, False]:
        for item in transaction["itemset"]:
            fname = item.feature_name
            fval  = item.rule_format

            feature_dict = probability_table[fname][fval]
            count = feature_dict[label]

            num_bins = len(probability_table[fname])

            # laplace smoothing
            prob = (count + 1) / (label_counts[label] + num_bins)

            scores[label] += math.log(prob)

    return max(scores, key=scores.get)

def build_naive_bayesian_classifier(args):
    try:
        trainset = CommonUtils.load_dataset(args.trainset_infile, args.entropy_weights)
        threshold_map = Discretizer.best_thresholds_for_features(trainset, args.max_split_count, args.min_bin_frac, args.delta_cost)
        transactions = apply_thresholds(trainset, threshold_map)

        probability_table = {}
        label_counts = {True: 0, False: 0}

        for t in transactions:
            label = t["label"]
            label_counts[label] += 1
            
            for item in t["itemset"]:
                fname = item.feature_name
                fval  = item.rule_format
                
                if fname not in probability_table:
                    probability_table[fname] = {}
                if fval not in probability_table[fname]:
                    probability_table[fname][fval] = {True: 0, False: 0}
                    
                probability_table[fname][fval][label] += 1

        out = {"probability_table" : probability_table, "threshold_map": threshold_map, "label_counts": label_counts}
        CommonUtils.save_pickle(out, args.pickle_path, "naive bayesian classifier probability table and threshold map")

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
        exit(0)

def evaluate_naive_bayesian_classifier(args):
    try:
        testset           = CommonUtils.load_dataset(args.testset_infile)
        pickled_data      = CommonUtils.load_pickle(args.pickle_path)

        probability_table = pickled_data["probability_table"]
        threshold_map     = pickled_data["threshold_map"]
        label_counts      = pickled_data["label_counts"]

        transactions =  apply_thresholds(testset, threshold_map)

        predictions  =  CommonHelpers.predict_dataset (
                            transactions, None,
                            probability_table, predict,
                            label_counts
                        )

        accuracy, precision, recall = CommonHelpers.get_basic_metrics([t["label"] for t in transactions], predictions)

        # TODO ROC AUC
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
        exit(0)
