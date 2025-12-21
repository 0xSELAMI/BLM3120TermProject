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
