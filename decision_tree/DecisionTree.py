# from common.Dataset import Dataset
from common.Utils import load_dataset, move_cursor_up_and_clear_line

import decision_tree.DecisionTreeHelpers as helpers
import decision_tree.TreeBuilder as tree_builder

from decision_tree.TreeNode import TreeNode

def predict_instance(node, instance):
    if node.is_leaf:
        return node.prediction

    feature_typename = node.feature_type.name

    val = getattr(instance, feature_typename)

    child_node = None

    if node.is_categorical:
        child_node = node.children.get(val)

        # shouldn't happen
        if child_node is None:
            return node.prediction
    else:
        child_node = node.children.get("right")

        if val <= node.threshold:
            child_node = node.children.get("left")

    return predict_instance(child_node, instance)

def predict_prob_instance(root_node, instance):
    node = root_node
    while not node.is_leaf:
        val = getattr(instance, node.feature_type.name)

        if node.is_categorical:
            child = node.children.get(val)

            if child is None:
                break

            node = child
        else:
            if val <= node.threshold:
                node = node.children["left"]
            else:
                node = node.children["right"]

    if node.prediction == True:
        npred = node.n_pred 
    else:
        npred = node.n_samples - node.n_pred 
        
    return npred / node.n_samples if node.n_samples > 0 else 0.0

def get_label_values_and_probs(root_node, dataset):
    label_name = list(dataset.feature_types.keys())[-1]

    y_labels = []
    y_probs = []

    for instance in dataset.instances:
        y_labels.append(1 if getattr(instance, label_name) else 0)
        y_probs.append(predict_prob_instance(root_node, instance))

    return (y_labels, y_probs)

def calc_roc_auc(y_label_values, y_probs):
    pairs = sorted(zip(y_probs, y_label_values), key=lambda x: x[0])

    ranks = range(1, len(pairs) + 1)
    rank_sum = 0
    ctx_label_true = 0

    for rank_val, (_, label_value) in zip(ranks, pairs):
        if label_value == True:
            rank_sum += rank_val
            ctx_label_true += 1

    ctx_label_false = len(pairs) - ctx_label_true

    # all same class
    if ctx_label_true == 0 or ctx_label_false == 0:
        return 0.0

    auc = (rank_sum - ctx_label_true * (ctx_label_true + 1) / 2) / (ctx_label_true * ctx_label_false)
    return auc


def predict_dataset(root_node, dataset):
    predictions = []

    for instance in dataset.instances:
        predictions.append(predict_instance(root_node, instance))

    return predictions

def build_decision_tree(args):
    tree_builder.MAX_DEPTH         = args.max_depth
    tree_builder.MIN_SAMPLES_SPLIT = args.min_samples_split
    tree_builder.MIN_GAIN          = args.min_info_gain
    tree_builder.USE_GINI          = args.use_gini

    helpers.MIN_SAMPLES_LEAF       = args.min_samples_leaf
    helpers.MIN_SAMPLES_LEAF_KARY  = args.min_samples_leaf_kary

    trainset = load_dataset(args.trainset_infile, args.entropy_weights)

    print("Building decision tree...", end = '\n\n')
    root = tree_builder.build_tree(trainset)
    move_cursor_up_and_clear_line(2)
    print("Building decision tree... Completed Successfully")
    print("Collapsing pure subtrees into leaves...")
    tree_builder.collapse_pure_subtrees(root)

    helpers.export_tree_to_dot(root, args.dot_outfile)
    helpers.pickle_decision_tree(root, args.pickle_path)

def evaluate_decision_tree(args):
    testset  = load_dataset(args.testset_infile)

    root = helpers.load_pickled_decision_tree(args.pickle_path)

    predictions = predict_dataset(root, testset)

    feature_names = list(testset.feature_types.keys())

    confusion_matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}

    for i, instance in enumerate(testset.instances):
        label = getattr(instance, feature_names[-1])

        if label == predictions[i]:
            if predictions[i] == True:
                confusion_matrix["TP"] += 1
            else:
                confusion_matrix["TN"] += 1
        else:
            if predictions[i] == True:
                confusion_matrix["FP"] += 1
            else:
                confusion_matrix["FN"] += 1

    preds_all_positive = confusion_matrix["TP"] + confusion_matrix["FP"]
    preds_tp_fn = confusion_matrix["TP"] + confusion_matrix["FN"]
    preds_tp_tn = confusion_matrix["TP"] + confusion_matrix["TN"]

    accuracy  = (preds_tp_tn / testset.size)
    precision = (confusion_matrix["TP"] / preds_all_positive)
    recall    = (confusion_matrix["TP"] / preds_tp_fn)
    f1_score  = 2 * ( (precision * recall) / (precision + recall) )

    roc_auc = calc_roc_auc(*get_label_values_and_probs(root, testset))

    print(f"Accuracy: %{round(accuracy*100, 4)} ({preds_tp_tn}/{testset.size})")

    print(f"True Positives: {confusion_matrix['TP']}/{testset.size}")
    print(f"True Negatives: {confusion_matrix['TN']}/{testset.size}")
    print(f"False Positives: {confusion_matrix['FP']}/{testset.size}")
    print(f"False Negatives: {confusion_matrix['FN']}/{testset.size}")

    print(f"Precision: %{round(precision * 100, 4)} ({confusion_matrix['TP']}/{preds_all_positive})")
    print(f"Recall: %{round(recall * 100, 4)} ({confusion_matrix['TP']}/{preds_tp_fn})")
    print(f"F1-Score: {round(f1_score, 4)}")
    print(f"ROC-AUC: {round(roc_auc, 4)}")
