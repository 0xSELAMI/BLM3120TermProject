import common.Utils as CommonUtils
import common.Helpers as CommonHelpers

import decision_tree.DecisionTreeHelpers as DecisionTreeHelpers
import decision_tree.TreeBuilder as TreeBuilder

from decision_tree.TreeNode import TreeNode

def predict(instance, node):
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

    return predict(instance, child_node)

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

def build_decision_tree(args):
    TreeBuilder.MAX_DEPTH         = args.max_depth
    TreeBuilder.MIN_SAMPLES_SPLIT = args.min_samples_split
    TreeBuilder.MIN_GAIN          = args.min_info_gain
    TreeBuilder.USE_GINI          = args.use_gini

    DecisionTreeHelpers.MIN_SAMPLES_LEAF       = args.min_samples_leaf
    DecisionTreeHelpers.MIN_SAMPLES_LEAF_KARY  = args.min_samples_leaf_kary

    trainset = CommonUtils.load_dataset(args.trainset_infile, args.entropy_weights)

    if not trainset:
        return None

    print("Building decision tree...", end = '\n\n')

    try:
        root = TreeBuilder.build_tree(trainset)
        CommonUtils.move_cursor_up_and_clear_line(2)
        print("Building decision tree... Completed Successfully")
        print("Collapsing pure subtrees into leaves...")
        TreeBuilder.collapse_pure_subtrees(root)

        DecisionTreeHelpers.export_tree_to_dot(root, args.dot_outfile)
        CommonUtils.save_pickle(root, args.pickle_path, "decision tree")

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
        return None

def evaluate_decision_tree(args):
    testset          = CommonUtils.load_dataset(args.testset_infile)

    if not testset:
        return None

    root             = CommonUtils.load_pickle(args.pickle_path)

    if not root:
        return None

    predictions      =  CommonHelpers.predict_dataset(
                            testset, lambda testset: testset.instances,
                            root, predict
                        )

    feature_names    = list(testset.feature_types.keys())

    confusion_matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}

    accuracy, precision, recall = CommonHelpers.get_basic_metrics([getattr(instance, feature_names[-1]) for instance in testset.instances], predictions)

    values_and_probs =  CommonHelpers.get_label_values_and_probs (
                            root, testset,
                            lambda testset: testset.instances,
                            lambda instance: getattr(instance, feature_names[-1]),
                            predict_prob_instance
                        )

    roc_auc = CommonHelpers.calc_roc_auc(*values_and_probs)

    print(f"ROC-AUC: {round(roc_auc, 4)}")
