import os
import sys
from enum import Enum

import git

sys.path.append(os.getcwd())

import argparse
import shutil
import pickle
import json
import numpy as np

from workflow.data_preparation import *
from git import Repo

# from eli5 import explain_prediction_xgboost, formatters


main_root = os.getcwd()
root = main_root + "/backend/"
csv_file = root + "data.csv"
model_file = root + "model_actual.dat"
original_data_file = main_root + '/data/data_headers.csv'

with open(model_file, "rb") as f:
    model = pickle.load(f)
    model.get_booster().set_param('nthread', 1)

name_mapping, reverse_mapping = load_encoder(main_root + '/data/encoder_actual.txt')
link_mapping, link_rmapping = load_encoder(main_root + '/data/encoder_links.txt', dtype=str)

cached_queries = {}

status_label = 'status'
result_label = 'result'
max_count = 100
batch_size = 10


class Status:
    queued = 'Queued for processing'
    loading = 'Loading repository'
    factorization = 'Computing features on source code'
    predictions = 'Making predictions, {}/{} finished'
    failed = 'Failed'
    success = 'Success'


def init_status(link):
    cached_queries[link] = { status_label: Status.queued }


def print_to_log(s):
    # f = open("log.txt", "a+")
    # f.writelines(s + '\n')
    # f.close()
    # sys.stderr.write(s + '\n')
    pass


def collect_features(path_to_project, path_to_csv, root, link):
    cached_queries[link][status_label] = Status.factorization

    print_to_log("Collecting features...")
    output = os.popen("java -jar {} {} {}".format(root + "run_coan", path_to_project, path_to_csv)).read()
    print_to_log(output)


def load_repo(link, path_to_csv, root):
    cached_queries[link][status_label] = Status.loading

    project_name = link.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    path_to_project = root + project_name

    print_to_log("Cloning repository {} from {}...".format(project_name, link))
    try:
        Repo.clone_from(link, path_to_project)
    except git.GitCommandError:
        cached_queries[link][status_label] = Status.failed
        raise ValueError('Invalid repository address')
    print_to_log("Cloned.")

    collect_features(path_to_project, path_to_csv, root, link)

    try:
        shutil.rmtree(path_to_project)
    except OSError as e:
        print_to_log("Error: %s - %s." % (e.filename, e.strerror))


def load_model_data(path_to_csv, path_to_original_data, link):
    data = load_data(path_to_csv)
    if len(data) == 0:
        cached_queries[link][status_label] = Status.failed
        raise ValueError("No java files found in repository")

    data, result_encoder = normalize_data(data)
    existing_features = data.columns.values

    print_to_log("Adding missing features...")
    original_data = load_data(path_to_original_data)
    original_data, original_encoder = normalize_data(original_data)
    fieldnames = original_data.columns.values
    for feature in fieldnames:
        if feature not in existing_features:
            data[feature] = 0
    data = data[fieldnames]
    print_to_log("Added.")
    print_to_log("Dropping rare features...")
    data = drop_rare_features(data)
    print_to_log("Dropped.")
    x, y = split_data(data)
    return x


def make_prediction(model, x, reverse_mapping, link):
    cached_queries[link][status_label] = Status.predictions.format(0, len(x))

    # print_to_log(formatters.format_as_text(explain_prediction_xgboost(model.get_booster(), x.iloc[0], top_targets=counts)))
    probs = None

    for s in range(0, len(x), batch_size):
        f = min(s+batch_size, len(x))
        batch_probs = model.predict_proba(x[s:f])
        if probs is None:
            probs = batch_probs
        else:
            probs = np.concatenate((probs, batch_probs), axis=0)
        cached_queries[link][status_label] = Status.predictions.format(f, len(x))

    resulting_probability = sum(probs) / len(x)

    class_probs = []
    for class_id, prob in zip(model.classes_, resulting_probability):
        # class_probs.append((class_id, prob))
        class_probs.append((reverse_mapping[class_id], prob))
    class_probs = sorted(class_probs, key=lambda x: x[1])

    return class_probs


def run_backend(link, counts):
    if link in cached_queries:
        status = cached_queries[link][status_label]

    if link in cached_queries and status != Status.failed:
        if status == Status.success:
            probabilities = cached_queries[link][result_label]
        else:
            return json.dumps({'status': status, 'error': ''})
    else:
        init_status(link)
        load_repo(link, csv_file, root)
        x = load_model_data(csv_file, original_data_file, link)

        print_to_log("Making predictions...")
        probabilities = make_prediction(model, x, reverse_mapping, link)
        cached_queries[link][result_label] = probabilities[-max_count:]
        print_to_log("Done.")
        cached_queries[link][status_label] = Status.success

    result = {'similarity': [], 'error': '', status_label: Status.success}
    print_to_log("\n--------------------\n")
    for project_name, prob in reversed(probabilities[-min(counts, len(probabilities)):]):
        print_to_log("{} -> {}".format(project_name, prob / probabilities[-1][1]))
        result['similarity'].append({
            'project': project_name,
            'rating': str(round(prob / probabilities[-1][1], 2)),
            'gitLink': link_mapping[project_name]
        })
    print_to_log("\n--------------------\n")
    return json.dumps(result)


def get_status(link):
    if link in cached_queries:
        return cached_queries[link][status_label]
    else:
        return Status.queued


def run_evaluation():
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("github_link", help="link to clone the project")
    args = parser.parse_args()
    run_backend(args.github_link, 10)
