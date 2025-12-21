import os
import pickle
from collections import Counter

from common.Utils import move_cursor_up_and_clear_line

import CAR.Discretizer as Discretizer

from CAR.Transaction import TransactionItem, TransactionItemset

def best_thresholds_for_features(dataset, max_split_count, min_bin_frac, delta_cost):
    threshold_map = {}

    print(f"Discretizing features, min_bin_frac: {min_bin_frac}, delta_cost: {delta_cost}\n")

    for feature_name, feature_type in dataset.feature_types.items():
        if feature_type.is_numeric:
            threshold_map[feature_name] = Discretizer.best_thresholds_for_feature(dataset, feature_name, max_split_count, min_bin_frac, delta_cost)

    return threshold_map

def apply_thresholds(dataset, threshold_map):
    transactions = []

    for instance in dataset.instances:
        items = []

        # without the last feature type, because thats the label
        feature_types = list(dataset.feature_types.values())

        label = feature_types[-1]

        feature_types = feature_types[:-1]

        for i, feature_type in enumerate(feature_types):
            is_placed = False

            feature_name = feature_type.name
            value = getattr(instance, feature_name)

            if feature_type.is_numeric:
                tmap = threshold_map[feature_name]

                for j, threshold in enumerate(tmap):
                    if value <= threshold:
                        if j == 0:
                            items.append(TransactionItem(feature_name, f"{feature_name} <= {threshold}"))
                            #items.append(f"{feature_name} <= {threshold}")
                        else:
                            items.append(TransactionItem(feature_name, f"{threshold} < {feature_name} <= {tmap[j]}"))
                            #items.append(f"{threshold} < {feature_name} <= {tmap[j]}")
                        
                        is_placed = True
                        break

                if not is_placed:
                    items.append(TransactionItem(feature_name, f"{feature_name} > {tmap[-1]}"))
                    #items.append(f"{feature_name} > {tmap[-1]}")

            else:
                items.append(TransactionItem(feature_name, f"{feature_name} = {value}"))
                #items.append(f"{feature_name} = {value}")

        itemset = TransactionItemset(items)
        transactions.append((itemset, getattr(instance, label.name)))

    return transactions

def get_F1(transactions, min_support):
    item_counts = Counter()

    for itemset, _ in transactions:
        for item in itemset:
            item_counts[item] += 1

    return {TransactionItemset([item]): count for item, count in item_counts.items() if (count / len(transactions)) >= min_support}

def generate_candidates(F_prev, k):
    candidates = TransactionItemset()
    itemsets = list(F_prev.keys())

    infostr = "Iterating through itemset "
    setcount = len(itemsets)

    for i in range(setcount):
        print(infostr + f"{i}/{setcount}")
        for j in range(i + 1, setcount):
            candidate = itemsets[i] | itemsets[j]

            if len(candidate) == k:
                candidates.add(candidate)

        move_cursor_up_and_clear_line(1)

    return candidates

def prune_candidates(candidates, F_prev):
    pruned = TransactionItemset()
    
    for candidate in candidates:
        if all( (candidate - {item}) in TransactionItemset(F_prev.keys()) for item in candidate ):
            pruned.add(candidate)

    return pruned

# how many a candidate itemse is found in the transactions list
def calc_candidate_counts(candidates, transactions, min_support):
    counts = {candidate : 0 for candidate in candidates}

    for items, _ in transactions:
        for candidate in candidates:
            if candidate.issubset(items):
                counts[candidate] += 1

    return {candidate: count for candidate, count in counts.items() if (count / len(transactions) >= min_support)}

def apriori(transactions, min_support, max_k):
    print("Collecting frequent itemsets with size 1")
    F = [get_F1(transactions, min_support)]

    k = 2

    infostr = f"Collecting frequent itemsets with size"

    while F[k - 2] and k <= max_k:
        move_cursor_up_and_clear_line(1)

        print(infostr + f" {k} : generating candidates")
        candidates_k = generate_candidates(F[k - 2], k)
        move_cursor_up_and_clear_line(1)

        print(infostr + f" {k} : pruning candidates")
        candidates_k = prune_candidates(candidates_k, F[k - 2])
        move_cursor_up_and_clear_line(1)

        print(infostr + f" {k} : counting candidates")
        Fk = calc_candidate_counts(candidates_k, transactions, min_support)

        if not Fk:
            break
        
        k += 1
        F.append(Fk)

    return F
