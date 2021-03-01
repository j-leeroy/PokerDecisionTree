import pandas as pd
import matplotlib.pyplot as plt
import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from six import StringIO

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read in the dataset
    test = pd.read_csv(r'C:\Users\jlgar\PycharmProjects\PokerDecisionTree\poker-hand-testing.data', header=None)
    test.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    # convert the Label into a binary classification using the map() function
    binaryLabel = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1,
                   5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    test['Label'] = test['Label'].map(binaryLabel)

    # same process read in the training dataset
    train = pd.read_csv(r'C:\Users\jlgar\PycharmProjects\PokerDecisionTree\poker-hand-training-true.data')
    train.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    train['Label'] = train['Label'].map(binaryLabel)

    # remove the Label class from the dataframe to only contain attributes
    X_train = train.loc[:, train.columns != 'Label']
    X_test = test.loc[:, test.columns != 'Label']

    # create the classification
    Y_train = train['Label']
    Y_test = test['Label']

    # create the decision tree object and learn a model
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train, Y_train)

    # using our new learned model make a prediction on our testing dataset
    y_pred = clf.predict(X_test)

    print('accuracy score of decision tree: ', accuracy_score(Y_test, y_pred))
    acc_score_tree = accuracy_score(Y_test, y_pred)

    # to visualize the decision tree, it will be saved in PDF format
    # this will take several minutes to execute
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True,
                    feature_names=list(X_train.columns),
                    class_names=['No Hand', 'Something in Hand'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')

    # Now creating an Adaboost object that will run for 15 iterations
    Ada = AdaBoostClassifier(n_estimators=15)
    # learn a model and then use it on our testing dataset
    model = Ada.fit(X_train, Y_train)
    y_pred_Ada = model.predict(X_test)

    print("Accuracy with Adaboost: ", accuracy_score(Y_test, y_pred_Ada))
    acc_score_ada = accuracy_score(Y_test, y_pred_Ada)

    # graphing stuff
    x = ['Decision Tree', 'Adaboost']
    y = [acc_score_tree, acc_score_ada]

    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, y)
    plt.ylabel("Accuracy Percent")
    plt.title("Accuracy Comparison")
    plt.xticks(x_pos, x)
    bars = plt.bar(x, height=y)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)

    plt.show()
