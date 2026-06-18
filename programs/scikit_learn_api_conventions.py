"""Runnable programs that demonstrate common Scikit-learn API conventions.

Run with:
    python programs/scikit_learn_api_conventions.py
"""

import numpy as np
from sklearn import random_projection
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.svm import SVC


def show_type_casting():
    """Show that a transformer can cast float32 input to float64 output."""
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2000).astype("float32")

    transformer = random_projection.GaussianRandomProjection(random_state=0)
    X_new = transformer.fit_transform(X)

    print("Type casting")
    print("Input dtype:", X.dtype)
    print("Transformed dtype:", X_new.dtype)
    print()


def show_refitting_and_parameter_updates():
    """Update estimator parameters with set_params() and fit again."""
    X, y = load_iris(return_X_y=True)
    clf = SVC()

    linear_prediction = clf.set_params(kernel="linear").fit(X, y).predict(X[:5])
    rbf_prediction = clf.set_params(kernel="rbf", gamma="scale").fit(X, y).predict(X[:5])

    print("Refitting and updating parameters")
    print("Linear kernel prediction:", linear_prediction)
    print("RBF kernel prediction:", rbf_prediction)
    print()


def show_multiclass_fitting():
    """Fit OneVsRestClassifier with a one-dimensional multiclass target."""
    X = [[1, 2], [3, 4], [4, 5], [5, 2], [1, 1]]
    y = [0, 0, 1, 1, 2]
    classifier = OneVsRestClassifier(estimator=SVC(gamma="scale", random_state=0))

    print("Multiclass fitting")
    print(classifier.fit(X, y).predict(X))
    print()


def show_binary_indicator_and_multilabel_fitting():
    """Fit with two-dimensional binary indicator and multilabel targets."""
    X = [[1, 2], [3, 4], [4, 5], [5, 2], [1, 1]]
    y = [0, 0, 1, 1, 2]
    classifier = OneVsRestClassifier(estimator=SVC(gamma="scale", random_state=0))

    y_binary = LabelBinarizer().fit_transform(y)
    binary_prediction = classifier.fit(X, y_binary).predict(X)

    y_multilabel = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
    y_multilabel = MultiLabelBinarizer().fit_transform(y_multilabel)
    multilabel_prediction = classifier.fit(X, y_multilabel).predict(X)

    print("Binary label indicator fitting")
    print(binary_prediction)
    print()
    print("Multilabel fitting")
    print(multilabel_prediction)


def main():
    show_type_casting()
    show_refitting_and_parameter_updates()
    show_multiclass_fitting()
    show_binary_indicator_and_multilabel_fitting()


if __name__ == "__main__":
    main()
