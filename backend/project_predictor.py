import os
import sys

sys.path.append(os.getcwd())

import argparse
import shutil
import pickle
import csv
import numpy as np

from workflow.data_preparation import *
from git import Repo


def collect_features(path_to_project, path_to_csv):
    print("Collecting features...")
    output = os.popen("java -jar {} {} {}".format(root + "coan", path_to_project, path_to_csv)).read()
    print(output)


def load_repo(link, path_to_csv):
    project_name = link.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    path_to_project = root + project_name

    print("Cloning repository {}...".format(project_name))
    Repo.clone_from(link, path_to_project)
    print("Cloned.")

    collect_features(path_to_project, path_to_csv)

    try:
        shutil.rmtree(path_to_project)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def load_model(path_to_csv, path_to_model, path_to_original_data):
    data = load_data(path_to_csv)
    data, result_encoder = normalize_data(data)
    existing_features = data.columns.values

    print("Adding missing features...")
    with open(path_to_original_data, 'r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        for feature in fieldnames:
            if feature not in existing_features:
                data[feature] = 0
        data = data[fieldnames]
    print("Added.")
    data = drop_rare_features(data)
    x, y = split_data(data)
    loaded_model = pickle.load(open(path_to_model, "rb"))
    return x, loaded_model


def make_prediction(model, x):
    probs = np.array(model.predict_proba(x))
    print(probs.shape)
    resulting_probability = sum(probs) / len(x)

    class_probs = []
    for class_id, prob in zip(model.classes_, resulting_probability):
        class_probs.append((reverse_mapping[class_id], prob))
    class_probs = sorted(class_probs, key=lambda x: x[1])

    return class_probs


if __name__ == '__main__':
    main_root = os.getcwd()
    root = main_root + "/backend/"
    csv_file = root + "data.csv"
    model_file = root + "model.dat"
    original_data_file = main_root + '/data/normalized_data.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument("github_link", help="link to clone the project")
    args = parser.parse_args()

    load_repo(args.github_link, csv_file)
    x, model = load_model(csv_file, model_file, original_data_file)
    name_mapping, reverse_mapping = load_encoder(main_root + '/data/result_encoder.txt')

    probabilities = make_prediction(model, x)

    print("\n--------------------\n")
    for project_name, prob in probabilities[-10:]:
        print("{} -> {}".format(project_name, prob))
    print("\n--------------------\n")

