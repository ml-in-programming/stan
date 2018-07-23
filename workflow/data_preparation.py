import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random

classes = 'Project'
random_seed = 239


def load_data(filename, index_col=None, header=0):
    print("Loading data from {}...".format(filename))

    data = pd.read_csv(filename, header=header, index_col=index_col, squeeze=True)

    print("Data loaded")

    return data


def load_encoder(filename, dtype=int):
    print("Loading encoder from {}...".format(filename))

    name_mapping = {}
    reverse_mapping = {}
    fin = open(filename, "r")
    for line in fin.readlines():
        string, result = line.split()
        result = dtype(result)
        name_mapping[string] = result
        reverse_mapping[result] = string

    print("Data loaded")

    return name_mapping, reverse_mapping


def normalize_data(data):
    print("Normalizing data...")

    int_features = ['Method', 'Fields', 'LocalVariables']
    nominal_features = []
    for feature in data.columns.values:
        if feature.startswith("Nominal"):
            nominal_features.append(feature)

    to_add = []
    for feature in nominal_features:
        values = data[feature].unique()
        new_features = list(map(lambda s: "Encoded" + feature + s, values))
        arrs = [[] for _ in range(len(values))]
        for index, row in data.iterrows():
            for value, arr in zip(values, arrs):
                if row[feature] == value:
                    arr.append(1)
                else:
                    arr.append(0)
        to_add.append((new_features, arrs))

    for feature in int_features:
        le = preprocessing.MinMaxScaler()
        data[feature] = le.fit_transform(data[[feature]])

    for new_features, arrs in to_add:
        for col_name, col_values in zip(new_features, arrs):
            data[col_name] = pd.Series(col_values)

    result_encoder = preprocessing.LabelEncoder().fit(data[classes])
    data[classes] = result_encoder.transform(data[classes])

    print("Data normalized")
    return data, result_encoder


def split_data(data, with_path=False):
    print("Splitting data...")

    to_drop = []
    for feature in data.columns.values:
        if feature.startswith("Nominal"):
            to_drop.append(feature)
    to_drop.append('Path')
    to_drop.append(classes)
    x = data.drop(to_drop, axis=1)
    y = data[classes] if not with_path else data[[classes, 'Path']]

    print("Data split")

    return x, y


def save_data(data, filename, index=False):
    print("Saving data to {0}...".format(filename))

    data.to_csv(filename, index=index)

    print("Saved")


def save_encoder(encoder, filename):
    print("Saving data to {0}...".format(filename))

    name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    fout = open(filename, "w")
    for string, result in name_mapping.items():
        fout.write(string + " " + str(result) + "\n")
    fout.close()

    print("Saved")


def drop_edge_classes(data, min_threshold, max_threshold=10000000):
    print("Dropping classes with population lower than {0} or greater than {1}...".format(min_threshold, max_threshold))

    counts = data[classes].value_counts()
    leave = []
    for index, row in data.iterrows():
        leave.append(min_threshold <= counts[row[classes]] <= max_threshold)
    data = data[leave]

    print("Dropped")

    return data


def take_fixed_samples(data, sizes):
    print("Finding classes with fixed sizes...")

    counts = data[classes].value_counts()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    #     print(counts)

    taken_class = [False] * len(counts)
    n_taken = [0] * len(sizes)
    leave = []
    cnt = 0
    for index, row in data.iterrows():
        found = False
        cls = row[classes]
        if taken_class[cls]:
            leave.append(True)
            continue

        for k in range(len(sizes)):
            if sizes[k][0] > n_taken[k] and sizes[k][1] == counts[cls]:
                n_taken[k] += 1
                taken_class[cls] = True
                cnt += 1
                found = True
                break
        leave.append(found)
    data = data[leave]
    print("Found {} classes".format(cnt))

    return data


def shuffle_data(data):
    return data.sample(frac=1)


def drop_rare_features(data):
    to_drop = ['RatioLambda', 'AstTypeAvgDepthexpr.MethodReferenceExpr',
               'AstTypeTFexpr.MethodReferenceExpr', 'AstTypeAvgDepthbody.AnnotationDeclaration',
               'AstTypeTFbody.AnnotationDeclaration', 'AstTypeAvgDepthbody.AnnotationMemberDeclaration',
               'AstTypeTFbody.AnnotationMemberDeclaration', 'AstTypeTFPackageDeclaration',
               'AstTypeAvgDepthstmt.LocalClassDeclarationStmt', 'AstTypeTFstmt.LocalClassDeclarationStmt',
               'AstTypeAvgDepthCompilationUnit', 'AstTypeAvgDepthstmt.DoStmt', 'AstTypeTFstmt.DoStmt',
               'AstTypeAvgDepthstmt.LabeledStmt', 'AstTypeTFstmt.LabeledStmt', 'AstTypeTFstmt.EmptyStmt',
               'AstTypeTFexpr.NormalAnnotationExpr', 'AstTypeAvgDepthexpr.MemberValuePair',
               'AstTypeTFexpr.MemberValuePair', 'AstTypeAvgDepthtype.UnknownType', 'AstTypeTFtype.UnknownType',
               'AstTypeAvgDepthexpr.TypeExpr', 'AstTypeTFexpr.TypeExpr', 'AstTypeAvgDepthtype.UnionType',
               'AstTypeTFtype.UnionType', 'AstTypeAvgDepthbody.InitializerDeclaration',
               'AstTypeTFbody.InitializerDeclaration', 'AstTypeAvgDepthstmt.AssertStmt', 'AstTypeTFstmt.AssertStmt',
               'AstTypeAvgDepthstmt.WhileStmt', 'AstTypeTFImportDeclaration', 'AstTypeTFexpr.LongLiteralExpr',
               'AstTypeAvgDepthexpr.LambdaExpr', 'AstTypeTFexpr.LambdaExpr',
               'EncodedNominalPunctuationBeforeBraceNewLine', 'EncodedNominalTabsLeadLinesTabs',
               'AstTypeTFbody.EnumDeclaration', 'AstTypeAvgDepthstmt.ContinueStmt', 'AstTypeAvgDepthstmt.SwitchStmt',
               'AstTypeAvgDepthexpr.CharLiteralExpr']

    return data.drop(to_drop, axis=1, errors='ignore')


def get_equal_samples(data, test_size, size=None, with_path=False):
    print("Picking random samples from all classes...")

    data = drop_rare_features(data)

    class_labels = data[classes].unique()
    if size is not None:
        class_labels = class_labels[:size]

    train_index = []
    test_index = []
    for n, label in enumerate(class_labels):
        sample = list(data.query('{0} == {1}'.format(classes, label)).index)
        train_sample, test_sample = train_test_split(sample, test_size=test_size, random_state=random_seed)
        train_index += train_sample
        test_index += test_sample
        # train = train.append(train_sample)
        # test = test.append(test_sample)

    train = data.ix[train_index]
    test = data.ix[test_index]
    print("classes:", len(class_labels))
    print("train:", len(train))
    print("test:", len(test))
    print("Samples are ready")

    x_train, y_train = split_data(train, with_path)
    x_test, y_test = split_data(test, with_path)
    return x_train, x_test, y_train, y_test


def with_buckets(data, num_buckets=10, size=None):
    print("Adding buckets to all classes...")

    # data = drop_rare_features(data)

    class_labels = data[classes].unique()
    if size is not None:
        class_labels = class_labels[:size]

    bucket_num, index = [], []
    for n, label in enumerate(class_labels):
        sample = list(data.query('{0} == {1}'.format(classes, label)).index)
        random.shuffle(sample)
        for i, id in enumerate(sample):
            bucket_num.append(i % num_buckets)
            index.append(id)

    print("Classes: {}".format(len(class_labels)))
    data['Bucket'] = pd.Series(bucket_num, index=index)
    print("Buckets are added")

    return data
