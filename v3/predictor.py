
#For SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#For decision tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def draw_decision_tree(X_train, y_train, feature_names=None, class_names=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None):
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)

    clf.fit(X_train, y_train)
    
    plt.figure(figsize=(20,10))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
    plt.show()


def do_SVM(X_train, X_test, y_train, y_test) : 
    
    svm_model = SVC(kernel='linear', random_state=42)

    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)


def main():
    return 0

if __name__ == '__main__':
    main()