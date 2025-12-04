from Dataset import Dataset
from Features import FeatureType, FeatureFilter

# FIXME this shouldn't be hardcoded
MIN_LEAF = 10

def calc_candidate_thresholds(dataset, feature_type):
    if not feature_type.is_numeric:
        raise ValueError(f"non-numeric feature type supplied to candidate threshold calculation: {feature_type}")

    unique_values = set()

    for instance in dataset.instances:
        unique_values.add(getattr(instance, feature_type.name))

    unique_values = sorted(unique_values)

    candidates = []

    for i in range(len(unique_values) - 1):
        candidates.append( (unique_values[i] + unique_values[i + 1]) / 2 )

    return candidates

def calc_info_gain_on_binary_split(parentset, feature_filter):
    left = Dataset.subset_with_feature_filter(parentset, [feature_filter])
    right = Dataset.subset_with_feature_filter(parentset, [feature_filter._not()])

    if left.is_empty or right.is_empty:
        return 0.0
    if left.size < MIN_LEAF or right.size < MIN_LEAF:
        return 0.0

    w_left = left.size / parentset.size
    w_right = right.size / parentset.size

    weighted_child_entropy = w_left * left.entropy + w_right * right.entropy

    return parentset.entropy - weighted_child_entropy

def calc_info_gain_on_kary_split(parentset, feature_type):
    value_domain = None

    if parentset.value_domains is not None:
        value_domain = parentset.value_domains[feature_type.name]
    else:
        value_domain = feature_type.value_domain

    subsets = []

    weighted_child_entropy = 0.0

    if type(feature_type.value_domain) == set:
        for value in value_domain:
            subset = Dataset.subset_with_feature_filter(parentset, [feature_type == value])

            if subset.size == 0:
                continue

            if subset.size < MIN_LEAF:
                return 0.0

            weighted_child_entropy += (subset.size / parentset.size) * subset.entropy
    else:
        raise ValueError(f"calc_info_gain_on_kary_split called with a non-set value domain: {feature_type}")

    return parentset.entropy - weighted_child_entropy

