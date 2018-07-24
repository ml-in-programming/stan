import os
import sys

sys.path.append(os.getcwd())

import argparse
import shutil
import pickle
import json
import numpy as np

from workflow.data_preparation import *
from git import Repo
from eli5 import explain_prediction_xgboost, formatters


def collect_features(path_to_project, path_to_csv, root):
    print("Collecting features...")
    output = os.popen("java -jar {} {} {}".format(root + "run_coan", path_to_project, path_to_csv)).read()
    print(output)


def load_repo(link, path_to_csv, root):
    project_name = link.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    path_to_project = root + project_name

    print("Cloning repository {}...".format(project_name))
    Repo.clone_from(link, path_to_project)
    print("Cloned.")

    collect_features(path_to_project, path_to_csv, root)

    try:
        shutil.rmtree(path_to_project)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def load_model(path_to_csv, path_to_model, path_to_original_data):
    data = load_data(path_to_csv)
    if len(data) == 0:
        raise ValueError("No java files found in repository")

    data, result_encoder = normalize_data(data)
    existing_features = data.columns.values

    print("Adding missing features...")
    original_data = load_data(path_to_original_data)
    original_data, original_encoder = normalize_data(original_data)
    fieldnames = original_data.columns.values
    for feature in fieldnames:
        if feature not in existing_features:
            data[feature] = 0
    data = data[fieldnames]
    print("Added.")
    data = drop_rare_features(data)
    x, y = split_data(data)
    loaded_model = pickle.load(open(path_to_model, "rb"))
    return x, loaded_model


def make_prediction(model, x, reverse_mapping, counts):
    # print(formatters.format_as_text(explain_prediction_xgboost(model.get_booster(), x.iloc[0], top_targets=counts)))
    probs = np.array(model.predict_proba(x))
    resulting_probability = sum(probs) / len(x)

    class_probs = []
    for class_id, prob in zip(model.classes_, resulting_probability):
        # class_probs.append((class_id, prob))
        class_probs.append((reverse_mapping[class_id], prob))
    class_probs = sorted(class_probs, key=lambda x: x[1])

    return class_probs


def run_backend(link, counts):
    main_root = os.getcwd()
    root = main_root + "/backend/"
    csv_file = root + "data.csv"
    model_file = root + "model_actual.dat"
    original_data_file = main_root + '/data/data_headers.csv'

    load_repo(link, csv_file, root)
    x, model = load_model(csv_file, model_file, original_data_file)

    name_mapping, reverse_mapping = load_encoder(main_root + '/data/encoder_actual.txt')
    link_mapping, link_rmapping = load_encoder(main_root + '/data/encoder_links.txt', dtype=str)

    probabilities = make_prediction(model, x, reverse_mapping, counts)

    result = {'similarity': [], 'error': ''}
    # print("\n--------------------\n")
    for project_name, prob in reversed(probabilities[-min(counts, len(probabilities)):]):
        # print("{} -> {}".format(project_name, prob / probabilities[-1][1]))
        result['similarity'].append({
            'project': project_name,
            'rating': str(round(prob / probabilities[-1][1], 2)),
            'gitLink': link_mapping[project_name]
        })
    # print("\n--------------------\n")
    return json.dumps(result)


def run_evaluation():
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("github_link", help="link to clone the project")
    args = parser.parse_args()
    run_backend(args.github_link, 10)
