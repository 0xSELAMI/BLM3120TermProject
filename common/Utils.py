import os
import re
import csv
import json
import pickle

from common.Dataset import DatasetSchema, Dataset
from sklearn.model_selection import train_test_split

import common.Logger as CommonLogger

# dataset field types ( excluding the first field which is user id )
field_types = [str, int, str, str, int, int, float, str, int, bool, bool]

def move_cursor_up_and_clear_line(times):
    cursorup = '\033[F'
    clear    = '\033[K'

    for _ in range(times):
        print(cursorup+clear, end='')

def process_dataset(args):
    dataset = load_dataset(args.dataset)

    if not dataset:
        return None

    if args.ratio <= 0 or args.ratio >= 1:
        CommonLogger.logger.log(f"[ERROR] Invalid testset to dataset ratio supplied: {args.ratio}, must be in range (0,1)")
        return None

    testset, trainset = create_test_and_train_set(dataset, args.ratio)

    save_dataset(args.trainset_outfile, testset)
    save_dataset(args.testset_outfile, trainset)

    CommonLogger.logger.log("[INFO] Test and Train datasets successfully created")
    yield

def save_dataset(file_path, dataset):
    out_path = os.path.normpath(file_path)
    directory, filename = os.path.split(out_path)

    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    with open(out_path, "w+") as outfile:
        json.dump(dataset, outfile)

def field_exists_in_dict(_dict, field):
    if (
        (not field in _dict) or
        (_dict[field] == None) or
        (len(_dict[field]) == 0)
    ):
        return False

    return True

def verify_dataset_format(dataset_contents):
    return (field_exists_in_dict(dataset_contents, "instances") and
            field_exists_in_dict(dataset_contents, "field_names"))

def load_dataset(dataset_filepath, entropy_weights = [1.0, 1.0]):
    global field_types

    if not dataset_filepath:
        CommonLogger.logger.log("[ERROR] load_dataset: dataset_filepath cannot be None")
        return None

    norm_path = os.path.normpath(dataset_filepath)
    directory, filename = os.path.split(norm_path)

    filename_noext, file_ext = os.path.splitext(filename)

    dataset_contents = None

    try:
        if file_ext == '.csv':
            with open(dataset_filepath) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                dataset_contents = list(reader)

                if not dataset_contents:
                    CommonLogger.logger.log("[ERROR] Empty dataset file")
                    return None
                elif len(dataset_contents) <= 1:
                    CommonLogger.logger.log("[ERROR] Dataset doesn't have more than a single line")
                    return None
                elif len(dataset_contents[1]) < len(field_types):
                    CommonLogger.logger.log("[ERROR] Dataset instance length doesn't match field types")
                    return None

                DatasetSchema.configure_schema(dataset_contents, field_types, entropy_weights)

                first_instance = dataset_contents[1][1:]

                for i, field in enumerate(first_instance):
                    try:
                        field_types[i](field)
                    except:
                        CommonLogger.logger.log("[ERROR] Dataset isn't compatible with the supplied field types")
                        return None

                dataset = Dataset([instance[1:] for instance in dataset_contents[1:]])

        elif file_ext == '.json':
            dataset_contents = json.load(open(dataset_filepath))

            if not verify_dataset_format(dataset_contents):
                CommonLogger.logger.log("[ERROR] Malformed dataset")
                return None

            DatasetSchema.configure_schema(dataset_contents, field_types, entropy_weights)

            dataset = Dataset(dataset_contents['instances'])
        elif file_ext == '':
            CommonLogger.logger.log(f"[ERROR] Dataset file extension should be '.json' or '.csv'")
            return None
        else:
            CommonLogger.logger.log(f"[ERROR] Supplied dataset file has unsupported extension: {filename}")
            return None
    except FileNotFoundError as e:
        CommonLogger.logger.log(f"[ERROR]{re.sub(r'\[Errno [0-9]+\]', '', str(e))}")
        return None

    return dataset

def create_test_and_train_set(dataset, ratio):
    x = []

    field_names = list(dataset.feature_types.keys())

    for i in dataset.instances:
        x.append([getattr(i, name) for name in field_names[:-1]])

    y = [getattr(i, field_names[-1]) for i in dataset.instances]

    # returns -> x train, x test, y train, y test
    # 1186 active, 414 churned in test instances (25.9%) churned
    # 4743 active, 1657 churned in training instances (25.6%) churned
    split = train_test_split(x, y, train_size = 1 - ratio, test_size = ratio, stratify=y)

    trainset = { "instances": [], "field_names": field_names }
    testset = { "instances": [], "field_names": field_names }

    for i in range(len(split[0])):
        trainset["instances"].append( split[0][i] + [split[2][i]] )

    for i in range(len(split[1])):
        testset["instances"].append( split[0][i] + [split[2][i]] )

    return (testset, trainset)

def save_pickle(data, pickle_outfile, datatype):
    out_path = os.path.normpath(pickle_outfile)
    directory, filename = os.path.split(out_path)

    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    with open(out_path, "wb+") as f:
        pickle.dump(data, f)

    CommonLogger.logger.log(f"Pickled {datatype} into file: {out_path}")
    yield

def load_pickle(pickle_infile):
    data = None

    try:
        with open(pickle_infile, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError as e:
        CommonLogger.logger.log(f"[ERROR]{re.sub(r'\[Errno [0-9]+\]', '', str(e))}")
        return None

    return data
