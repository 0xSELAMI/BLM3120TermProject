import os
import pickle

from common.Dataset import Dataset
from common.Helpers import calc_candidate_thresholds
from common.Logger import logger

MIN_SAMPLES_LEAF = None
MIN_SAMPLES_LEAF_KARY = None

def calc_info_gain_on_binary_split(parentset, feature_filter, use_gini=False):
    right = Dataset.subset_with_feature_filter(parentset, [feature_filter])
    left = Dataset.subset_with_feature_filter(parentset, [feature_filter._not()])

    if left.is_empty or right.is_empty:
        return 0.0
    if left.size < MIN_SAMPLES_LEAF or right.size < MIN_SAMPLES_LEAF:
        return 0.0

    w_left = left.size / parentset.size
    w_right = right.size / parentset.size

    gain = None

    if (use_gini):
        weighted_child_gini = w_left * left.gini + w_right * right.gini
        gain = parentset.gini - weighted_child_gini
    else:
        weighted_child_entropy = w_left * left.entropy + w_right * right.entropy
        gain = parentset.entropy - weighted_child_entropy
    
    return gain

def calc_info_gain_on_kary_split(parentset, feature_type, use_gini=False):
    value_domain = None

    if parentset.value_domains is not None:
        value_domain = parentset.value_domains[feature_type.name]
    else:
        value_domain = feature_type.value_domain

    subsets = []

    weighted_child_values = 0.0

    if type(feature_type.value_domain) == set:
        for value in value_domain:
            subset = Dataset.subset_with_feature_filter(parentset, [feature_type == value])

            if subset.size == 0:
                continue

            if subset.size < MIN_SAMPLES_LEAF_KARY:
                return 0.0

            if use_gini:
                weighted_child_values += (subset.size / parentset.size) * subset.gini
            else:
                weighted_child_values += (subset.size / parentset.size) * subset.entropy
    else:
        raise ValueError(f"calc_info_gain_on_kary_split called with a non-set value domain: {feature_type}")

    gain = None

    if use_gini:
        gain = parentset.entropy - weighted_child_values
    else:
        gain = parentset.gini - weighted_child_values

    return gain

def evaluate_info_gains(dataset, use_gini=False):
    best_gain = 0.0
    best_split = None

    feature_names = list(dataset.feature_types.keys())
    label_name = feature_names[-1]

    for fname in feature_names[:-1]:
        ftype = dataset.feature_types[fname]

        if ftype.is_numeric:
            thresholds = calc_candidate_thresholds(dataset, ftype)
            for T in thresholds:
                gain = calc_info_gain_on_binary_split(dataset, ftype > T, use_gini)
                if gain > best_gain:
                    best_gain = gain
                    best_split = ("numeric", ftype, T)
        else:
            # categorical
            gain = calc_info_gain_on_kary_split(dataset, ftype, use_gini)
            if gain > best_gain:
                best_gain = gain
                best_split = ("categorical", ftype, None)

    return (best_gain, best_split)

def export_tree_to_dot(root, dot_outfile):
    lines = []
    lines.append("digraph DecisionTree {")
    lines.append("  rankdir=TB;")  # vertical layout
    lines.append("  node [shape=box, fontsize=10, height=0.3, width=0.3];")
    lines.append("  edge [fontsize=9];")

    node_id_counter = [0]

    def next_id():
        node_id_counter[0] += 1
        return f"n{node_id_counter[0]}"

    def escape(text):
        return text.replace('"', '\\"')

    def add_node(node):
        node_id = next_id()

        if node.is_leaf:
            pred = escape(str(node.prediction))
            label = f"{pred}\\n({node.n_pred}/{node.n_samples})"
            lines.append(f'  {node_id} [label="{label}", shape=ellipse];')
            return node_id

        feat_name = escape(node.feature_type.name)
        label = f"{feat_name}\\nn = {node.n_samples}"

        lines.append(f'  {node_id} [label="{label}"];')

        if node.is_categorical:
            for val, child in node.children.items():
                child_id = add_node(child)
                lines.append(
                    f'  {node_id} -> {child_id} [label="{escape(str(val))}"];'
                )

        else:  # numeric
            thr = node.threshold
            left = node.children.get("left")
            right = node.children.get("right")

            if left:
                left_id = add_node(left)
                lines.append(
                    f'  {node_id} -> {left_id} [label="≤ {thr}"];'
                )
            if right:
                right_id = add_node(right)
                lines.append(
                    f'  {node_id} -> {right_id} [label="> {thr}"];'
                )

        return node_id

    add_node(root)
    lines.append("}")

    out_path = os.path.normpath(dot_outfile)
    directory, filename = os.path.split(out_path)
 
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
 
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    logger.log(f"Exported tree as DOT to {out_path}")
    yield
