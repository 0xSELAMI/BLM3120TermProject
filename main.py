#!/usr/bin/env python

import os
import re
import csv
import sys
import json
import argparse
import numpy as np

from sklearn.metrics import entropy
from sklearn.model_selection import train_test_split

class Instance:
    def __init__(self, *args):
        if len(args) != 11:
            print('[ERROR] Invalid instance initialization')
            return None

        self.gender = args[0]
        self.age = args[1]
        self.country = args[2]
        self.subscription_type = args[3]
        self.listening_time = args[4]
        self.songs_played_per_day = args[5]
        self.skip_rate = args[6]
        self.device_type = args[7]
        self.ads_listened_per_week = args[8]
        self.offline_listening = args[9]
        self.is_churned = args[10]
        
    def __repr__(self):
        return f"""Gender: {self.gender}
Age: {self.age}
Country: {self.country}
Subscription Type: {self.subscription_type}
Listening Time: {self.listening_time} Minutes
Songs Played Per Day: {self.songs_played_per_day}
Skip Rate: {self.skip_rate}
Device Type: {self.device_type}
Ads Listened Per Week: {self.ads_listened_per_week}
Offline Listening: {'True' if self.offline_listening else 'False'}
Is Churned: {'True' if self.is_churned else 'False'}"""

def save_dataset(file_path, instances):
    out_path = os.path.normpath(file_path)
    directory, filename = os.path.split(out_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    dataset = { 'instances': [] }

    with open(out_path, "w+") as outfile:
        for instance in instances:
            dataset['instances'].append(instance)

        json.dump(dataset, outfile)

def load_dataset(dataset_filepath):
    norm_path = os.path.normpath(dataset_filepath)
    directory, filename = os.path.split(norm_path)

    filename_noext, file_ext = os.path.splitext(filename)

    dataset_contents = None

    try:
        if file_ext == '.csv':
            with open(dataset_filepath) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                dataset_contents = list(reader)
        elif file_ext == '.json':
            dataset_contents = json.load(open(dataset_filepath))
        elif file_ext == '':
            print(f"[ERROR] Dataset file extension should be '.json' or '.csv'")
            exit(1)
        else:
            print(f"[ERROR] Supplied dataset file has unsupported extension: {filename}")
            exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR]{re.sub(r'\[Errno [0-9]+\]', '', str(e))}")
        #print(f"[ERROR] {e.__class__.__name__} {getattr(e, 'message', e)}")
        exit(1)

    return dataset_contents

def create_test_and_train_set(dataset_contents, ratio):
    all_instances = []

    for data in dataset_contents[1:]:
        instance = { "feature_vector": data[1:-1], "label": int(data[-1]) }
        all_instances.append(instance)

    x = [i["feature_vector"] for i in all_instances]
    y = [i["label"] for i in all_instances]

    # returns -> x train, x test, y train, y test
    # 1186 active, 414 churned in test instances (25.9%) churned
    # 4743 active, 1657 churned in training instances (25.6%) churned
    split = train_test_split(x, y, train_size = 1 - ratio, test_size = ratio, stratify=y)

    train_instances = []
    test_instances = [] 

    for i in range(len(split[0])):
        train_instances.append({ "feature_vector": split[0][i], "label": split[2][i] })

    for i in range(len(split[1])):
        test_instances.append({ "feature_vector": split[1][i], "label": split[3][i] })
    
    return (test_instances, train_instances)

def process_dataset(args):
    dataset_contents = load_dataset(args.dataset)
    test_instances, train_instances = create_test_and_train_set(dataset_contents, args.ratio)

    save_dataset(args.trainset_outfile, test_instances)
    save_dataset(args.testset_outfile, train_instances)

def main():
    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = """
                            A python script to oversee and fulfill the functionalities
                            the project proposal document specifies
                        """,
                        epilog='StudentID: [REDACTED]')

    subparsers = parser.add_subparsers(title="commands", dest="command")

    parser_process_dataset = subparsers.add_parser("process_dataset", description="Create testset and trainset from supplied dataset", help="Create testset and trainset from supplied dataset")

    parser_process_dataset.add_argument("--dataset", "-d", metavar='PATH_TO_DATASET', type=str, required=True)

    parser_process_dataset.add_argument("--ratio", "-r", metavar='RATIO', help="ratio of the size of the testset to the size of whole dataset (default: 0.2)", default=0.2, type=float)

    parser_process_dataset.add_argument("--trainset-outfile", metavar='TRAINSET_OUTPATH', help="default: dataset/trainset.json", default="dataset/trainset.json", type=str)

    parser_process_dataset.add_argument("--testset-outfile", metavar='TESTSET_OUTPATH', help="default: dataset/testset.json", default="dataset/testset.json", type=str)

    args = parser.parse_args()

    if args.command == "process_dataset":
        process_dataset(args)
    elif args.command == "x":
        pass
    else:
        parser.print_help()

    # TODO: gotta

if __name__ == "__main__":
    main()
