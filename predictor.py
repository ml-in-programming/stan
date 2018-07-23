import os
import argparse
from workflow.data_preparation import *
from workflow.models.xgb import train_sklearn_xgb_classifier, train_xgb_clf, xgb_train_buckets
from workflow.models.clustering import kmeans_clustering
from array import array


def concatenate_data(path_to_data, data_prefix, start, end):
    dfs = []
    for i in range(start, end):
        dfs.append(load_data(data_prefix + "data{}.csv".format(i)))
    data = pd.concat(dfs)
    save_data(data, path_to_data)


def normalize(path_to_data, path_to_normalized_data, path_to_encoder):
    data = load_data(path_to_data)
    data, result_encoder = normalize_data(data)
    save_data(data, path_to_normalized_data)
    save_encoder(result_encoder, path_to_encoder)


def create_dataset(path_to_normalized_data, test_size, size, min_files, max_files, prefix):
    data = load_data(path_to_normalized_data)
    data = drop_edge_classes(data, min_files, max_files)
    x_train, x_test, y_train, y_test = get_equal_samples(data, test_size=test_size, size=size, with_path=True)

    save_data(x_train, prefix + 'x_train.csv', index=True)
    save_data(y_train, prefix + 'y_train.csv', index=True)
    save_data(x_test, prefix + 'x_test.csv', index=True)
    save_data(y_test, prefix + 'y_test.csv', index=True)


def add_buckets(path_to_normalized_data, prefix):
    data = load_data(path_to_normalized_data)
    data = take_fixed_samples(data, [(11, 31), (5, 32), (1, 210), (1, 212)])
    data = with_buckets(data, 10)
    save_data(data, prefix + 'bucketed_data.csv')


def train_and_test(path_to_result, path_to_classifier, path_to_encoder, prefix):
    name_mapping, reverse_mapping = load_encoder(path_to_encoder)
    x_train = load_data(prefix + 'x_train.csv', index_col=0)
    y_train = load_data(prefix + 'y_train.csv', index_col=0)
    x_test = load_data(prefix + 'x_test.csv', index_col=0)
    y_test = load_data(prefix + 'y_test.csv', index_col=0)

    # results = train_xgb_clf(x_train, y_train, x_test, y_test, target='Project', nthread=4)
    results = train_sklearn_xgb_classifier(x_train, y_train, x_test, y_test, path_to_classifier=path_to_classifier,
                                           target='Project', nthread=4)
    # y_test['Correct'] = pd.Series(results, index=y_test.index)
    # save_data(y_test, prefix + 'y_test.csv', index=True)
    # if path_to_result:
    #     output_file = open(path_to_result, 'wb')
    #     float_array = array('d', results)
    #     float_array.tofile(output_file)
    #     output_file.close()


def train_buckets(prefix):
    data = load_data(prefix + 'bucketed_data.csv')
    xgb_train_buckets(data, target='Project', bucket_target='Bucket', n_buckets=10, nthread=4)


def cluster(path_to_encoder, prefix):
    name_mapping, reverse_mapping = load_encoder(path_to_encoder)
    x_train = load_data(prefix + 'x_train.csv', index_col=0)
    y_train = load_data(prefix + 'y_train.csv', index_col=0)
    x_test = load_data(prefix + 'x_test.csv', index_col=0)
    y_test = load_data(prefix + 'y_test.csv', index_col=0)
    results = kmeans_clustering(x_train, y_train, x_test, y_test, reverse_mapping)


if __name__ == '__main__':
    root = os.getcwd() + "/"
    prefix_subsample = root + 'data/subsample/'
    prefix_buckets = root + 'data/buckets/'

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="choose between \"normalize\", \"create_dataset\", \"cluster\" and \"train\"")
    parser.add_argument("-s", "--size", help="number of classes used for evaluation in prediction", type=int)
    parser.add_argument("-ts", "--test_size", help="part of classes used for validation in prediction", type=float)
    parser.add_argument("-d", "--data", help="relative path to data in csv format")
    parser.add_argument("-nd", "--normalized_data", help="relative path to normalized data in csv format")
    parser.add_argument("-le", "--encoder", help="relative path to result encoder")
    parser.add_argument("-sm", "--save_model", help="relative path to trained model")
    parser.add_argument("-mf", "--min_files", help="minimal number of files in project", type=int, default=10)
    parser.add_argument("-Mf", "--max_files", help="maximal number of files in project", type=int, default=100)
    parser.add_argument("--save_result", help="store result in file", action='store_true')
    args = parser.parse_args()

    if args.mode == "concatenate":
        path_to_data = root + "data/data.csv"
        if args.data:
            path_to_data = root + args.data
        concatenate_data(path_to_data, root + "../rawData/", 0, 9)

    elif args.mode == "normalize":
        path_to_data = root + "data/data.csv"
        path_to_normalized_data = root + "data/normalized_data.csv"
        path_to_encoder = root + "data/result_encoder.txt"
        if args.data:
            path_to_data = root + args.data
        if args.normalized_data:
            path_to_normalized_data = root + args.normalized_data
        if args.encoder:
            path_to_encoder = root + args.encoder
        normalize(path_to_data=path_to_data,
                  path_to_normalized_data=path_to_normalized_data,
                  path_to_encoder=path_to_encoder)

    elif args.mode == "create_dataset":
        path_to_normalized_data = root + "data/normalized_data.csv"
        test_size = 0.2
        if args.normalized_data:
            path_to_normalized_data = root + args.normalized_data
        if args.test_size:
            test_size = args.test_size

        create_dataset(path_to_normalized_data=path_to_normalized_data,
                       test_size=test_size,
                       size=args.size,
                       min_files=args.min_files,
                       max_files=args.max_files,
                       prefix=prefix_subsample)

    elif args.mode == "add_buckets":
        path_to_normalized_data = root + "data/normalized_data.csv"
        if args.normalized_data:
            path_to_normalized_data = root + args.normalized_data

        add_buckets(path_to_normalized_data=path_to_normalized_data,
                    prefix=prefix_buckets)

    elif args.mode == "train":
        path_to_encoder = root + "data/result_encoder.txt"
        path_to_result = None
        path_to_model = None
        if args.encoder:
            palth_to_encoder = root + args.encoder
        if args.save_result:
            path_to_result = root + "data/results_{}.out".format(args.size)
        if args.save_model:
            path_to_model = root + args.save_model

        # train_buckets(prefix_buckets)
        train_and_test(path_to_result=path_to_result,
                       path_to_classifier=path_to_model,
                       path_to_encoder=path_to_encoder,
                       prefix=prefix_subsample)

    elif args.mode == "cluster":
        path_to_encoder = root + "data/result_encoder.txt"
        if args.encoder:
            path_to_encoder = root + args.encoder
        cluster(path_to_encoder=path_to_encoder, prefix=prefix_subsample)
    else:
        print("invalid mode")
