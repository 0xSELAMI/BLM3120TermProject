import os
import pickle
from collections import Counter

from common.Logger import logger
import common.Utils as CommonUtils
import common.Discretizer as Discretizer

from common.Transaction import TransactionItemset

def get_F1(transactions, min_support):
    item_counts = Counter()

    for t in transactions:
        for item in t['itemset']:
            item_counts[item] += 1

    sorted_items = sorted(item_counts.items(), key=lambda x: str(x[0]))
    return {TransactionItemset([item]): count for item, count in sorted_items if (count / len(transactions)) >= min_support}

def generate_candidates(F_prev, k):
    candidates = set()
    itemsets = sorted(list(F_prev.keys()), key=lambda x: str(x))

    infostr = "Iterating through previous frequent itemsets... "
    setcount = len(itemsets)

    for i in range(setcount):
        itemset_i = itemsets[i]

        logger.log(infostr + f"{i}/{setcount}")
        yield

        for j in range(i + 1, setcount):
            candidate = itemset_i | itemsets[j]

            if len(candidate) == k:
                candidates.add(candidate)

        logger.backtrack(1)
        #CommonUtils.move_cursor_up_and_clear_line(1)

    return TransactionItemset(candidates)

def prune_candidates(candidates, F_prev):
    pruned = set()

    candidates = list(candidates)
    
    infostr = "Iterating through candidates... "
    candidate_count = len(candidates)

    for i, candidate in enumerate(candidates):
        logger.log(infostr + f"{i}/{candidate_count}")
        yield

        if all( (candidate - {item}) in set(F_prev.keys()) for item in candidate ):
            pruned.add(candidate)

        logger.backtrack(1)
        #CommonUtils.move_cursor_up_and_clear_line(1)

    return TransactionItemset(pruned)

# how many a candidate itemse is found in the transactions list
def calc_candidate_counts(candidates, transactions, min_support):
    counts = {candidate : 0 for candidate in candidates}

    infostr = "Iterating through transactions... "

    candidates = list(candidates.items)
    transaction_count = len(transactions)

    for i, t in enumerate(transactions):
        t_itemset = t["itemset"]

        logger.log(infostr + f"{i}/{transaction_count}")
        yield

        for candidate in candidates:
            if len(t_itemset) < len(candidate):
                continue

            if candidate.issubset(t_itemset):
                counts[candidate] += 1

        logger.backtrack(1)
        #CommonUtils.move_cursor_up_and_clear_line(1)

    return {candidate: count for candidate, count in counts.items() if (count / len(transactions) >= min_support)}

def apriori(transactions, min_support, max_k):
    logger.log("Collecting frequent itemsets with size 1")
    yield

    F = [get_F1(transactions, min_support)]

    k = 2

    infostr = f"Collecting frequent itemsets with size"

    while F[k - 2] and k <= max_k:
        logger.update_last(infostr + f" {k} : generating candidates")
        yield

        candidates_k = yield from generate_candidates(F[k - 2], k)

        logger.update_last(infostr + f" {k} : pruning candidates")
        yield

        candidates_k = yield from prune_candidates(candidates_k, F[k - 2])

        logger.update_last(infostr + f" {k} : counting candidate occurances in transactions")
        yield
        Fk = yield from calc_candidate_counts(candidates_k, transactions, min_support)

        if not Fk:
            break
        
        k += 1
        F.append(Fk)

    yield

    return F

def generate_rules(all_frequent_itemsets, transactions, min_support, min_confidence, min_lift):
    rules = []

    label_supports = {
        True:  sum(1 for t in transactions if t["label"]) / len(transactions),
        False: sum(1 for t in transactions if not t["label"]) / len(transactions)
    }

    infostr = "Generating CARs..."
    logger.log(infostr)

    for i, Fk in enumerate(all_frequent_itemsets):
        sorted_Fk = sorted(Fk.items(), key=lambda x: str(x[0]))
        logger.update_last(infostr + f" processing {i+1}-item frequent itemsets with {len(Fk)} itemsets")
        yield

        # for every itemset in the current frequent itemset
        for frequent_itemset, count_X in sorted_Fk:
            counts_X_y = {True: 0, False: 0}

            # keep track of counts of transactions with possible labels
            # if a transaction that contains current itemset
            # increase the count of that label
            for t in transactions:
                if frequent_itemset.issubset(t["itemset"]):
                    # increase the count of (X -> y) where X is the itemset in Fk
                    # and y is a label
                    counts_X_y[t["label"]] += 1

            max_conf = max(counts_X_y[True], counts_X_y[False]) / count_X

            if max_conf < min_confidence:
                continue

            for label, count_X_y in counts_X_y.items():
                # |transactions with itemset X and label y| / |transactions|
                support    = count_X_y / len(transactions)

                # |transactions with itemset X and label y| / |transactions with itemset X|
                confidence = count_X_y / count_X

                lift = confidence / label_supports[label]

                p = label_supports[label]
                m = (1 - label_supports[label]) / label_supports[label]
                m_estimate = (count_X_y + (m * p)) / (count_X + m)

                current_rule = {
                    "itemset":    frequent_itemset,
                    "label":      label,
                    "lift":       confidence / label_supports[label],
                    "m_estimate": m_estimate,
                    "confidence": confidence,
                    "support":    support
                }

                if current_rule["lift"] <= min_lift:
                    continue

                #This rule must be 50% more likely than a random guess for that class.
                if (
                    current_rule["m_estimate"] < (label_supports[label] * 2) and
                    current_rule["label"] == True
                ):
                    continue

                rules.append(current_rule)

        #CommonUtils.move_cursor_up_and_clear_line(1)

    return rules, label_supports

def build_classifier(rules, transactions, error_weights):
    def rule_covers(rule, idx):
        return rule["itemset"].issubset(transactions[idx]["itemset"])

    def rule_is_correct(rule, idx):
        return rule_covers(rule, idx) and (transactions[idx]["label"] == rule["label"])

    def rule_is_incorrect(rule, idx):
        return rule_covers(rule, idx) and (transactions[idx]["label"] != rule["label"])

    def count_labels_in_remainder(remaining_idx, transactions):
        count_true = sum(1 for i in remaining_idx if transactions[i]["label"])
        count_false = len(remaining_idx) - count_true

        return (count_true, count_false)

    def count_labels_in_transactions(transactions):
        count_true = sum(1 for t in transactions if t["label"])
        count_false = len(transactions) - count_true

        return (count_true, count_false)

    remaining_idx = set(range(len(transactions)))
    rule_list         = []
    total_errors      = []
    cumulative_errors = 0

    count_true_t, count_false_t = count_labels_in_transactions(transactions)

    count_true = count_false = None

    for rule in rules:
        default_errors = None

        covered = [i for i in remaining_idx if rule_covers(rule, i)]

        if not covered:
            continue

        correct = [i for i in remaining_idx if rule_is_correct(rule, i)]
        wrong   = [i for i in remaining_idx if rule_is_incorrect(rule, i)]
        
        # if the rule is more wrong than right then skip
        if not correct or len(wrong) >= len(correct):
            continue
        
        rule_list.append(rule)

        # cost of ignoring the incorrectly covered instances (false neg/pos)
        if rule["label"]:
            cumulative_errors += len(wrong) * error_weights[0]
        else:
            cumulative_errors += len(wrong) * error_weights[1]

        for i in covered:
            remaining_idx.remove(i)

        if remaining_idx:
            count_true, count_false = count_labels_in_remainder(remaining_idx, transactions)

            # cost of stopping construction and creating a default label
            if count_true >= count_false:
                # false instances we'd get wrong with a default label
                default_errors = count_false * error_weights[0]
            else:
                # true instances we'd get wrong with a default label
                default_errors = count_true * error_weights[1]
        else:
            default_errors = 0

        total_errors.append(cumulative_errors + default_errors)

        if not remaining_idx:
            break

    if not rule_list:
        count_true    = sum(1 for t in transactions if t["label"])
        count_false   = len(transactions) - count_true
        default_label = count_true >= count_false
        default_rule  = {"itemset": set(), "label": default_label, "default": True}

        return [], default_rule

    stopping_point = total_errors.index(min(total_errors))
    pruned_rules   = rule_list[:stopping_point + 1]

    # the first time, we used remaining_idx to get to the stopping point
    # now we use it to determine the default label
    remaining_idx = set(range(len(transactions)))

    for rule in pruned_rules:
        covered = [i for i in remaining_idx if rule_covers(rule, i)]
        for i in covered:
            remaining_idx.remove(i)

    if remaining_idx:
        count_true, count_false = count_labels_in_remainder(remaining_idx, transactions)
    else:
        count_true = sum(1 for t in transactions if t["label"])
        count_false = len(transactions) - count_true

    default_label = (count_true >= count_false)
    default_rule = {"itemset": set(), "label": default_label, "default": True}

    return pruned_rules, default_rule
