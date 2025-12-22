from common.Dataset import Dataset
from common.Utils import move_cursor_up_and_clear_line

def extract_thresholds(dataset, segment_map, split_count):
    thresholds = []

    values = [getattr(instance, dataset.sorted_on) for instance in dataset.instances]

    cur_seg_end = len(values) - 1
    bin_count   = split_count + 1

    # go backwards from segment end to start, get the threshold for the seg start
    # since the segment threshold covers instances until seg end
    # and decrease bin count while doing this,
    # since because every aggregation forms a bin
    while bin_count > 1:
        seg_start  = segment_map[bin_count][cur_seg_end]

        threshold = (values[seg_start] + values[seg_start + 1]) / 2
        thresholds.append(round(threshold, 6))

        cur_seg_end = seg_start
        bin_count -= 1

    return sorted(thresholds)

def discretize(dataset, split_count, min_frac=0.1):
    dataset.calc_positive_counts()
    desired_bin_count = split_count + 1

    # min amount of values a bin should have, to cover min_frac of all values
    MIN_BIN_SIZE = int(min_frac * dataset.size)

    if MIN_BIN_SIZE == 0:
        MIN_BIN_SIZE = 1

    # cost_map[b][i] = cost of using b bins to cover instances[0..i] (inclusive)
    cost_map = [[float("inf")] * dataset.size for _ in range(desired_bin_count + 1)]

    # segment_map[b][i] = index t where last bin is [t+1 .. i] (inclusive)
    segment_map = [[None] * dataset.size for _ in range(desired_bin_count + 1)]

    # base case
    for i in range(dataset.size):
        if i + 1 >= MIN_BIN_SIZE:
            cost_map[1][i] = dataset.calc_segment_cost(0, i)

    for bin_count in range(2, desired_bin_count + 1):
        for seg_end in range(dataset.size):

            # j should be splitable into b bins
            if seg_end + 1 < bin_count * MIN_BIN_SIZE:
                continue

            for seg_start in range(bin_count - 2, seg_end):

                if (seg_end - seg_start) < MIN_BIN_SIZE:
                    # not enough elements
                    continue

                if cost_map[bin_count - 1][seg_start] == float("inf"):
                    # can't have a segment without a starting point
                    continue

                if getattr(dataset.instances[seg_start], dataset.sorted_on) == getattr(dataset.instances[seg_start + 1], dataset.sorted_on):
                    # not a meaningful segmentation
                    continue

                cost = cost_map[bin_count - 1][seg_start] + dataset.calc_segment_cost(seg_start + 1, seg_end)

                if cost < cost_map[bin_count][seg_end]:
                    cost_map[bin_count][seg_end]    = cost
                    segment_map[bin_count][seg_end] = seg_start

    if cost_map[desired_bin_count][dataset.size - 1] == float("inf"):
        return None, None

    return cost_map[desired_bin_count][dataset.size - 1], segment_map

def best_thresholds_for_feature(trainset, feature_name, max_split_count, min_bin_frac, delta_cost):
    best_cost       = float("inf")
    best_thresholds = None

    for split_count in range(1, max_split_count + 1):
        outstr = f"feature: {feature_name}, trying split_count: {split_count}\n"

        dataset                          = Dataset.sort_on_feature(trainset, feature_name)
        discretization_cost, segment_map = discretize(dataset, split_count, min_bin_frac)
        thresholds                       = None

        if discretization_cost is not None:
            thresholds = extract_thresholds(dataset, segment_map, split_count)

            diff_cost = (best_cost - discretization_cost) if (discretization_cost < best_cost) else 0

            if diff_cost > delta_cost:
                best_cost       = discretization_cost
                best_thresholds = thresholds

        outstr += f"best_thresholds: {best_thresholds}\nbest_cost: {round(best_cost, 6)}"

        if split_count < max_split_count:
            print(outstr)
            move_cursor_up_and_clear_line(3)
        else:
            outstr = f"Discretized feature: {feature_name}, split_count: {len(best_thresholds)}\n"
            outstr += f"best_thresholds: {best_thresholds}\nbest_cost: {round(best_cost, 6)}"
            print(outstr + '\n')

    return best_thresholds
