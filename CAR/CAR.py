from common.Utils import load_dataset, move_cursor_up_and_clear_line, save_pickle, load_pickle
from common.Dataset import Dataset

import CAR.CARHelpers as helpers

def generate_CARs(args):
    trainset  = load_dataset(args.trainset_infile, args.entropy_weights)

    rules = []

    infostr = "Generating CARs..."

    try:
        threshold_map         = helpers.best_thresholds_for_features(trainset, args.max_split_count, args.min_bin_frac, args.delta_cost)
        transactions          = helpers.apply_thresholds(trainset, threshold_map)
        
        max_k          = args.max_k
        min_support    = args.min_support
        min_confidence = args.min_confidence

        print(f"Running apriori algorithm... (max_k: {max_k}, min_support: {min_support}, min_confidence: {min_confidence})")
        # apriori returns all Fk where k in range (0, max_k)
        all_frequent_itemsets = helpers.apriori(transactions, min_support, max_k)
        move_cursor_up_and_clear_line(2)
        print(f"Collected frequent itemsets up to size {len(all_frequent_itemsets)}. (max_k: {max_k}, min_support: {min_support}, min_confidence: {min_confidence})")


        # for every freq itemset
        for i, Fk in enumerate(all_frequent_itemsets):
            print(infostr + f" processing {i+1}-item frequent itemsets with {len(Fk)} itemsets")

            # for every itemset in the current frequent itemset
            for itemset, count_X in Fk.items():
                counts_X_y = {True: 0, False: 0}

                # keep track of counts of transactions with possible labels
                # if a transaction that contains current itemset
                # increase the count of that label
                for items, label in transactions:
                    if itemset.issubset(items):
                        # increase the count of (X -> y) where X is the itemset in Fk
                        # and y is a label
                        counts_X_y[label] += 1

                max_conf = max(counts_X_y[True], counts_X_y[False]) / count_X

                if max_conf < min_confidence:
                    continue

                # support and confidence
                sc = []

                for label, count_X_y in counts_X_y.items():
                    # |transactions with itemset X and label y| / |transactions|
                    support    = count_X_y / len(transactions)

                    # |transactions with itemset X and label y| / |transactions with itemset X|
                    confidence = count_X_y / count_X

                    sc.append({"confidence": confidence, "support": support, "label": label})

                sc.sort(key = lambda r : ( -r["confidence"], -r["support"] ))

                current_rule = {
                    "itemset": itemset,
                    "label": sc[0]["label"],
                    "confidence": sc[0]["confidence"],
                    "support": sc[0]["support"]
                }
                
                if not len(rules):
                    rules.append(current_rule)
                else:
                    rules.sort(key = lambda r : ( -r["confidence"], -r["support"], len(r["itemset"]) ))

                    rule_is_subsumed = False

                    for rule in rules:
                        if (
                            # check if a rule better than this one already exists
                            rule["itemset"].issubset(current_rule["itemset"]) and
                            current_rule["label"] == rule["label"] and
                            current_rule["confidence"] <= rule["confidence"] and
                            current_rule["support"] <= rule["support"]
                        ):
                            rule_is_subsumed = True
                            break
                        
                        if not rule_is_subsumed:
                            rules.append(current_rule)
                            break

            move_cursor_up_and_clear_line(1)

        # sort to have descending confidence, support and increasing itemset,
        # prioritized by value's order in the tuple
        rules.sort(key = lambda r : ( -r["confidence"], -r["support"], len(r["itemset"]) ))

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
        exit(0)

    print(f"Generated {len(rules)} rules.\n")

    save_pickle(rules, args.pickle_path, "class association rules")

def evaluate_CARs(args):
    pass
