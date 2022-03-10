"""
The Chinese University of Hong Kong, Shenzhen
CSC1001, Spring 2020
Instructor: Prof. Xiaoguang Han

Final Project
CART Classification Tree on Redwine Quality

Finished by:
    Chongyu Fang
    Shuhui Xiang
"""


import math
import pandas as pd
from time import time


class Node(object):
    """
    Define the Node class.
    """

    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None


def info_entropy(df, predict_attr):
    """
    Calculate entropy of the data.
    """

    # Number of high/low quality observations
    p_df = df[df[predict_attr] == 1]
    n_df = df[df[predict_attr] == 0]
    p = float(p_df.shape[0])
    n = float(n_df.shape[0])

    # Calculate entropy
    if p  == 0 or n == 0:
        I = 0
    else:
        I = ((-1 * p) / (p + n)) * math.log(p / (p + n), 2) + ((-1 * n)/(p + n)) * math.log(n / (p + n), 2)

    return I


def remainder(df, df_subsets, predict_attr):
    """
    Calculates the weighted average of the entropy after an attribute test.
    """

    num_data = df.shape[0]
    remainder = float(0)

    for df_sub in df_subsets:
        if df_sub.shape[0] > 1:
            remainder += float(df_sub.shape[0]/num_data)*info_entropy(df_sub, predict_attr)

    return remainder


def info_gain(df, attribute, predict_attr, threshold):
    """
    Calculates the information gain from the attribute
    test based on a given threshold. Note that thresholds
    can change for the same attribute over time.
    """

    sub_1 = df[df[attribute] < threshold]
    sub_2 = df[df[attribute] > threshold]

    # Determine information content and subract
    # remainder of attributes from it
    ig = info_entropy(df, predict_attr) - remainder(df, [sub_1, sub_2], predict_attr)

    return ig


def num_class(df, predict_attr):
    """
    Returns the number of positive and negative data.
    """

    p_df = df[df[predict_attr] == 1]
    n_df = df[df[predict_attr] == 0]

    return p_df.shape[0], n_df.shape[0]


def select_threshold(df, attribute, predict_attr):
    """
    Select the threshold of the attribute to split data.
    The threshold chosen splits the test data such that
    information gain is maximized.
    """

    values = df[attribute].tolist()
    values = [float(x) for x in values]
    values = list(set(values))
    values.sort()

    max_ig = float("-inf")
    thres_val = 0
    
    # Try all threshold values that are half-way between
    # successive values in this sorted list
    for i in range(0, len(values) - 1):
        thres = (values[i] + values[i+1]) / 2
        ig = info_gain(df, attribute, predict_attr, thres)
        if ig > max_ig:
            max_ig = ig
            thres_val = thres


    return thres_val


def choose_attr(df, attributes, predict_attr):
    """
    Chooses the attribute and its threshold with the
    highest info gain from the set of attributes.
    """

    max_info_gain = float("-inf")
    best_attr = None
    threshold = 0

    # Test each attribute (note attributes maybe be chosen more than once)
    for attr in attributes:
        thres = select_threshold(df, attr, predict_attr)
        ig = info_gain(df, attr, predict_attr, thres)
        if ig > max_info_gain:
            max_info_gain = ig
            best_attr = attr
            threshold = thres

    return best_attr, threshold


def build_tree(df, cols, predict_attr):
    """
    Builds the decision tree based on training data,
    attributes to train on, and a prediction attribute.
    """

    p, n = num_class(df, predict_attr)
    # If train data has all positive or all negative values
    # then we have reached the end of our tree
    if p == 0 or n == 0:
        # Create a leaf node indicating its prediction
        leaf = Node(None,None)
        leaf.leaf = True
        if p > n:
            leaf.predict = 1
        else:
            leaf.predict = 0

        return leaf

    # Otherwise recursively build the tree
    else:
        best_attr, threshold = choose_attr(df, cols, predict_attr)
        # Create internal tree node based on attribute and its threshold
        tree = Node(best_attr, threshold)
        sub_1 = df[df[best_attr] < threshold]
        sub_2 = df[df[best_attr] > threshold]
        # Recursively build left and right subtree
        tree.left = build_tree(sub_1, cols, predict_attr)
        tree.right = build_tree(sub_2, cols, predict_attr)

        return tree


def predict(node, row_df):
    """
    Given an instance of a training data, make a prediction
    of high quality or low quality based on the decision tree.
    Note that all data has been cleaned (i.e. no NULL data).
    """

    # If we are at a leaf node, return the prediction of the leaf node
    if node.leaf:
        return node.predict

    # Traverse left or right subtree based on instance's data
    if row_df[node.attr] <= node.thres:
        return predict(node.left, row_df)

    elif row_df[node.attr] > node.thres:
        return predict(node.right, row_df)


def test_predictions(root, df):
    """
    Given a set of data, make a prediction for each
    instance using the decision tree.
    """

    num_data = df.shape[0]
    num_correct = 0

    for index, row in df.iterrows():
        prediction = predict(root, row)
        if prediction == row['Outcome']:
            num_correct += 1

    return round(num_correct/num_data, 4)


def clean(csv_file_name):
    """
    Cleans the input data.
    Remove 'quality' column and add 'Outcome' column
    where 0 means low quality (quality score <= 6) and
    1 means high quality (quality score > 6).
    """

    df = pd.read_csv(csv_file_name, header = 0)
    df.columns = ['fixed acidity',
                    'volatile acidity',
                    'citric acid',
                    'residual sugar',
                    'chlorides',
                    'free sulfur dioxide',
                    'total sulfur dioxide',
                    'density',
                    'pH',
                    'sulphates',
                    'alcohol',
                    'quality']

    # Create new column 'Outcome' that assigns low quality
    # wine a value of 0 and high quality wine value of 1.
    df['Outcome'] = 0
    df.loc[df['quality'] > 6, 'Outcome'] = 1
    df.drop(['quality'], axis = 1)
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors = 'coerce')

    return df


def main():
    """
    Our main function.
    """

    df_train = clean("train.csv")
    attributes =  ['fixed acidity',
                    'volatile acidity',
                    'citric acid',
                    'residual sugar',
                    'chlorides',
                    'free sulfur dioxide',
                    'total sulfur dioxide',
                    'density',
                    'pH',
                    'sulphates',
                    'alcohol']

    root = build_tree(df_train, attributes, 'Outcome')
    df_test = clean("test.csv")

    accuracy = test_predictions(root, df_test) * 100.0
    display_str = "Accuracy of test data: {} %".format(accuracy)
    
    print(display_str)


if __name__ == '__main__':
    startTime = time()
    main()
    endTime = time()
    print("The algorithm of ID3 Tree costs", endTime-startTime, "seconds for predicting quality on the red wine dataset.")
