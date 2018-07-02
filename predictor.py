import os
import argparse
from workflow.data_preparation import *
from workflow.models.xgb import *
from array import array


def normalize(path_to_data, path_to_normalized_data, path_to_encoder):
    data = load_data(path_to_data)
    data, result_encoder = normalize_data(data)
    save_data(data, path_to_normalized_data)
    save_encoder(result_encoder, path_to_encoder)


def predict(path_to_normalized_data, path_to_encoder, test_size, size, min_files, path_to_result):
    data = load_data(path_to_normalized_data)
    name_mapping, reverse_mapping = load_encoder(path_to_encoder)
    data = drop_edge_classes(data, min_files)
    x_train, x_test, y_train, y_test = get_equal_samples(data, test_size=test_size, size=size)
    # train_xgb_clf(x_train, y_train, x_test, y_test)
    results = train_sklearn_xgb_classifier(x_train, y_train, x_test, y_test, full=True)
    if path_to_result:
        output_file = open(path_to_result, 'wb')
        float_array = array('d', results)
        float_array.tofile(output_file)
        output_file.close()


if __name__ == '__main__':
    root = os.getcwd() + "/"

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="choose between \"normalize\" and \"predict\"")
    parser.add_argument("-s", "--size", help="number of classes used for evaluation in prediction", type=int)
    parser.add_argument("-ts", "--test_size", help="part of classes used for validation in prediction", type=float)
    parser.add_argument("-d", "--data", help="relative path to data in csv format")
    parser.add_argument("-nd", "--normalized_data", help="relative path to normalized data in csv format")
    parser.add_argument("-le", "--encoder", help="relative path to result encoder")
    parser.add_argument("-mf", "--min_files", help="minimal number of files in project", type=int, default=10)
    parser.add_argument("--save_result", help="store result in file", action='store_true')
    args = parser.parse_args()

    if args.mode == "normalize":
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

    elif args.mode == "predict":
        path_to_normalized_data = root + "data/normalized_data.csv"
        path_to_encoder = root + "data/result_encoder.txt"
        path_to_result = None
        test_size = 0.2
        if args.normalized_data:
            path_to_normalized_data = root + args.normalized_data
        if args.encoder:
            path_to_encoder = root + args.encoder
        if args.test_size:
            test_size = args.test_size
        if args.save_result:
            path_to_result = root + "data/results_{}.out".format(args.size)
        predict(path_to_normalized_data=path_to_normalized_data,
                path_to_encoder=path_to_encoder,
                test_size=test_size,
                size=args.size,
                min_files=args.min_files,
                path_to_result=path_to_result)

    else:
        print("invalid mode")
