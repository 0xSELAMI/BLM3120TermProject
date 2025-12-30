from defaults import *

DISCRETIZER_FIELDS = [
    {"id": "entropy_weights", "label": "Entropy Weights", "info":"Entropy weights for true and false labels respectively", "value":','.join((str(default_entropy_weights)[1:-1]).split(", "))},
    {"id": "max_split_count", "label": "Max Split Count", "type": "number", "value": default_max_split_count, "info": "Max split count to consider while discretizing numeric features"},
    {"id": "min_bin_frac", "label": "Minimum Bin Fraction", "type": "number", "value": default_min_bin_frac, "info": "Minimum fraction of the training dataset a threshold bin should cover while discretizing numeric features into multiple bins"},
    {"id": "delta_cost", "label": "Delta Cost", "type": "number", "value": default_delta_cost, "info":"Minimum cost difference adding a new bin should make while discretizing numeric features into multiple bins"}
]

PREPROCESS_DATASET = {
    "title": "Preprocess Dataset",
    "handler": "process_dataset",
    "sections": [
        {
            "layout": "group",
            "fields": [
                {"id": "testset_outfile", "label": "Testset Output Path", "type": "path", "value": default_testset_path},
                {"id": "trainset_outfile", "label": "Trainset Output Path", "type": "path", "value": default_trainset_path},
                {
                    "id": "ratio", "label": "Testset/Dataset Ratio", "type": "number", "value": default_testset_trainset_ratio, "info": f"({default_testset_trainset_ratio} corresponds to trainset with {1 - default_testset_trainset_ratio} ratio)"
                },
            ]
        },
    ]
}

DECISION_TREE = {
    "title": "Decision Tree Classifier",
    "subtabs": [
        {
            "title": "Build",
            "handler": "decision_tree.build",
            "sections": [
                {
                    "layout": "row",
                    "sections": [
                        {
                            "layout": "column",
                            "sections": [
                                {
                                    "layout": "group",
                                    "fields": [
                                        {"id": "trainset_infile", "label": "Trainset Path", "type": "path", "value": default_trainset_path},
                                        {"id": "pickle_path", "label": "Pickle Path", "type": "path", "value": default_decision_tree_pickle_path, "info":"Path to pickle the decision tree into"},
                                        {"id": "dot_outfile", "label": "Dot Export Path", "type": "path", "value": default_dot_outfile, "info":"Path to write the dotfile of the decision tree to"},
                                        {"id": "use_gini", "label": "Use Gini", "info":"Use gini impurity instead of entropy", "type": "dropdown", "choices": [True, False], "value":False},
                                        {"id": "entropy_weights", "label": "Entropy Weights", "info":"Entropy weights for true and false labels respectively (useless if using gini)", "value":','.join((str(default_entropy_weights)[1:-1]).split(", "))}
                                    ]
                                }
                            ]
                        },
                        {
                            "layout": "column",
                            "sections": [
                                {
                                    "layout": "group",
                                    "fields": [
                                        {"id": "max_depth", "label": "Max Tree Depth", "type":"number", "value":default_max_depth, "info":"Max decision tree depth"},
                                        {"id": "min_info_gain", "label": "Minimum Split Info Gain", "type":"number", "value":default_min_info_gain, "info":"Minimum info gain for a split to qualify"},
                                        {"id": "min_samples_split", "label":"Minimum Split Sample Size", "type":"number", "value":default_min_samples_split, "info":"Minimum samples a meaningful split should have"},
                                        {"id": "min_samples_leaf", "label": "Minimum Leaf Sample Size", "type":"number", "value":default_min_samples_leaf, "info":"Minimum samples a leaf node should have"},
                                        {"id": "min_samples_leaf_kary", "label": "Minimum K-ary Leaf Sample Size", "type":"number", "value":default_min_samples_leaf_kary, "info":"Minimum samples a leaf node of a k-ary node should have"},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "title": "Evaluate",
            "handler": "decision_tree.evaluate",
            "sections": [
                {
                    "layout": "group",
                    "fields": [
                        {"id": "testset_infile", "label": "Testset Path", "type": "path", "value": default_testset_path},
                        {"id": "pickle_path", "label": "Pickle Path", "type": "path", "value": default_decision_tree_pickle_path, "info":"Path to read pickled decision tree from"},
                    ]
                }
            ]
        }
    ]
}

CBA = {
    "title": "CBA Classifier",
    "subtabs": [
        {
            "title": "Generate",
            "handler": "CBA.generate",
            "sections": [
                {
                    "layout": "row",
                    "sections": [
                        {
                            "layout": "column",
                            "sections": [
                                {
                                    "layout": "group",
                                    "fields": [
                                        {"id": "trainset_infile", "label": "Trainset Path", "type": "path", "value": default_trainset_path},
                                        {"id": "pickle_path", "label": "Pickle Path", "type": "path", "value": default_CBA_pickle_path, "info":"Path to pickle the classifier rules into"},
                                    ]
                                },
                                {
                                    "layout": "accordion",
                                    "label": "Discretization Params",
                                    "fields": DISCRETIZER_FIELDS
                                }
                            ]
                        },
                        {
                            "layout": "column",
                            "sections": [
                                {
                                    "layout": "group",
                                    "fields": [
                                        {"id": "max_k", "label": "Max K", "type":"number", "value":default_max_k, "info":"Max K value for apriori"},
                                        {"id": "min_supoprt", "label": "Minimum Support", "type":"number", "value":default_min_support, "info":"Minimum support for the CARs"},
                                        {"id": "min_confidence", "label": "Minimum Confidence", "type":"number", "value":default_min_confidence, "info":"Minimum confidence for the CARs"},
                                        {"id": "min_lift", "label": "Minimum Lift", "type":"number", "value":default_min_lift, "info":"Minimum lift for the CARs"},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "title": "Evaluate",
            "handler": "CBA.evaluate",
            "sections": [
                {
                    "layout": "group",
                    "fields": [
                        {"id": "testset_infile", "label": "Testset Path", "type": "path", "value": default_testset_path},
                        {"id": "pickle_path", "label": "Pickle Path", "type": "path", "value": default_CBA_pickle_path, "info":"Path to read pickled rules from"},
                    ]
                }
            ]
        }
    ]
}

NAIVE_BAYESIAN = {
    "title": "Naive Bayesian Classifier",
    "subtabs": [
        {
            "title": "Build",
            "handler": "naive_bayesian.build",
            "sections": [
                {
                    "layout": "row",
                    "sections": [
                        {
                            "layout": "column",
                            "sections": [
                                {
                                    "layout": "group",
                                    "fields": [
                                        {"id": "trainset_infile", "label": "Trainset Path", "type": "path", "value": default_trainset_path},
                                        {"id": "pickle_path", "label": "Pickle Path", "type": "path", "value": default_naive_bayesian_pickle_path, "info":"Path to pickle the probability table into"},
                                    ]
                                }
                            ]
                        },
                        {
                            "layout": "column",
                            "sections": [
                                {
                                    "layout": "accordion",
                                    "label": "Discretization Params",
                                    "fields": DISCRETIZER_FIELDS
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "title": "Evaluate",
            "handler": "naive_bayesian.evaluate",
            "sections": [
                {
                    "layout": "group",
                    "fields": [
                        {"id": "testset_infile", "label": "Testset Path", "type": "path", "value": default_testset_path},
                        {"id": "pickle_path", "label": "Pickle Path", "type": "path", "value": default_naive_bayesian_pickle_path, "info":"Path to read pickled probability table from"},
                    ]
                }
            ]
        }
    ]
}

layout_definition = [PREPROCESS_DATASET, DECISION_TREE, CBA, NAIVE_BAYESIAN]
