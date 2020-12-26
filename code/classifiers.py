from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score, roc_curve, \
    precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm


def show_roc(title, y_test, y_predict):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def get_lr_hyper_parameters(x_train, y_train):
    # hyperparameter tuning by RandomizedsearchCV
    hyperparameters = {
        'C': randint(0.0001, 1000),
        'penalty': ['l1', 'l2'],
        'max_iter': randint(100, 500),
        'class_weight': [{1: 0.5, 0: 0.5}, {1: 0.4, 0: 0.6}, {1: 0.6, 0: 0.4}, {1: 0.7, 0: 0.3}, {1: 0.8, 0: 0.2}]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = RandomizedSearchCV(estimator=LogisticRegression(solver="saga"), param_distributions=hyperparameters,
                             random_state=1, n_iter=10, cv=cv, verbose=0, n_jobs=-1)
    clf.fit(x_train, y_train)
    print('The Top Parameters Are:', clf.best_params_)
    print('The Top Parameter Score Is:', clf.best_score_)
    return clf.best_params_


def apply_lr(x_train, y_train, x_test, y_test, hyper_parameters):
    top_penalty = hyper_parameters['penalty']
    top_C = hyper_parameters['C']
    top_max_iter = hyper_parameters['max_iter']
    top_class_weight = hyper_parameters['class_weight']

    model = LogisticRegression(C=top_C, max_iter=top_max_iter, penalty=top_penalty, solver='saga', n_jobs=-1,
                               random_state=1, class_weight=top_class_weight)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print("Accuracy of logistic regression model is:", metrics.accuracy_score(y_test, prediction) * 100)
    conf_matrix = confusion_matrix(y_test, prediction)
    print(conf_matrix)
    print('the recall score is:', recall_score(y_test, prediction))
    print(classification_report(y_test, prediction))

    fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Logistic Regression Classifier')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return conf_matrix


def apply_lr(x_train, y_train, x_test, y_test):

    model = LogisticRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print("Accuracy of logistic regression model is:", metrics.accuracy_score(y_test, prediction) * 100)
    conf_matrix = confusion_matrix(y_test, prediction)
    print(conf_matrix)
    print('the recall score is:', recall_score(y_test, prediction))
    print(classification_report(y_test, prediction))

    fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Logistic Regression Classifier')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return conf_matrix


def get_dt_hyper_parameters(x_train, y_train):
    hyperparameters = {
        'min_samples_split': randint(2, 20),
        'max_depth': randint(100, 1000),
        'criterion': ["gini", "entropy"],
        'splitter': ['random', 'top'],
        'min_samples_leaf': randint(1, 20),
        'max_leaf_nodes': randint(2, 20)

    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = RandomizedSearchCV(DecisionTreeClassifier(), cv=cv, param_distributions=hyperparameters, n_iter=10)
    clf.fit(x_train, y_train)
    print('The Top Parameters Are:', clf.best_params_)
    print('The Top Parameter Score Is:', clf.best_score_)

    return clf.best_params_


def apply_dt(x_train, y_train, x_test, y_test, hyper_parameters):
    top_criterion = hyper_parameters['criterion']
    top_max_depth = hyper_parameters['max_depth']
    top_min_samples_leaf = hyper_parameters['min_samples_leaf']
    top_splitter = hyper_parameters['splitter']
    top_min_samples_split = hyper_parameters['min_samples_split']
    top_max_leaf_nodes = hyper_parameters['max_leaf_nodes']

    model = DecisionTreeClassifier(criterion=top_criterion,
                                   max_depth=top_max_depth,
                                   min_samples_leaf=top_min_samples_leaf,
                                   max_leaf_nodes=top_max_leaf_nodes,
                                   min_samples_split=top_min_samples_split,
                                   splitter=top_splitter, random_state=1)
    model = model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of decision tree classifier is:", metrics.accuracy_score(y_test, prediction) * 100)

    show_roc("Decision Tree", y_test, prediction)

    fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Decision Tree Classifier')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def apply_dt(x_train, y_train, x_test, y_test):

    model = DecisionTreeClassifier(random_state=1)
    model = model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of decision tree classifier is:", metrics.accuracy_score(y_test, prediction) * 100)

    show_roc("Decision Tree", y_test, prediction)

    fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Decision Tree Classifier')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def get_rf_hyper_parameters(x_train, y_train):
    hyperparameters = {

        'max_features': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': randint(1, 20),
        'n_estimators': randint(100, 500),
        'max_depth': randint(10, 100)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = RandomizedSearchCV(RandomForestClassifier(), cv=cv, param_distributions=hyperparameters, n_iter=10)
    clf.fit(x_train, y_train)
    print('The Top Parameters Are:', clf.best_params_)
    print('The Top Parameter Score Is:', clf.best_score_)

    return clf.best_params_

def apply_rf(x_train, y_train, x_test, y_test, hyperparameters):
    top_n_estimators = hyperparameters['n_estimators']
    top_max_features = hyperparameters['max_features']
    top_criterion = hyperparameters['criterion']
    top_min_samples_split = hyperparameters['min_samples_split']
    top_min_samples_leaf = hyperparameters['min_samples_leaf']
    top_max_depth = hyperparameters['max_depth']

    model = RandomForestClassifier(
        criterion=top_criterion,
        min_samples_split=top_min_samples_split,
        max_depth=top_max_depth,
        max_features=top_max_features,
        min_samples_leaf=top_min_samples_leaf,
        n_estimators=top_n_estimators,
        bootstrap=True)

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of random forest model is:", metrics.accuracy_score(y_test, prediction) * 100)

    show_roc("Random Forest", y_test, prediction)

def apply_rf(x_train, y_train, x_test, y_test):

    model = RandomForestClassifier(n_estimators=100, bootstrap=True)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of random forest model is:", metrics.accuracy_score(y_test, prediction) * 100)
    show_roc("Random Forest", y_test, prediction)


def get_mlp_hyper_parameters(x_train, y_train):
    hyperparameters = {
        'hidden_layer_sizes': [25, (25, 25), (40, 40), (25, 30, 25), (45, 40, 45), (65, 60, 65), (85, 80, 85)],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'activation': ['tanh', 'relu', 'logistic']

    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = RandomizedSearchCV(MLPClassifier(), cv=cv, param_distributions=hyperparameters, n_jobs=-1)
    clf.fit(x_train, y_train)
    print('The Top Parameters Are:', clf.best_params_)
    print('The Top Parameter Score Is:', clf.best_score_)

    return clf.best_params_


def apply_mlp(x_train, y_train, x_test, y_test, hyperparameters):
    top_hidden_layer_sizes = hyperparameters['hidden_layer_sizes']
    top_activation = hyperparameters['activation']
    top_solver = hyperparameters['solver']
    top_learning_rate = hyperparameters['learning_rate']

    model = MLPClassifier(hidden_layer_sizes=top_hidden_layer_sizes, max_iter=100,
                          activation=top_activation,
                          solver=top_solver, learning_rate=top_learning_rate)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of MLP model is:", metrics.accuracy_score(y_test, prediction) * 100)

    show_roc("Multi-Layer Perceptron", y_test, prediction)

def apply_mlp(x_train, y_train, x_test, y_test):

    model = MLPClassifier(max_iter=100)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of MLP model is:", metrics.accuracy_score(y_test, prediction) * 100)

    show_roc("Multi-Layer Perceptron", y_test, prediction)

def get_svm_hyper_parameters(x_train, y_train):
    hyperparameters = {
        "C": [0.001, 0.01,0.1,0.3,0.5,0.7,1,3,5,7,9],
        "gamma": randint(0.01, 1),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': randint(1, 10)

    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = RandomizedSearchCV(estimator=SVC(), cv=cv, param_distributions=hyperparameters, n_iter=20, n_jobs=4,
                             random_state=42)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    print('The Top Parameter Score Is:', clf.best_score_)

    return clf.best_params_


def apply_svm(x_train, y_train, x_test, y_test, hyperparameters):
    top_C = hyperparameters['C']
    top_gamma = hyperparameters['gamma']
    top_kernel = hyperparameters['kernel']
    top_degree = hyperparameters['degree']

    model = SVC(C=top_C, degree=top_degree, gamma=top_gamma, kernel=top_kernel)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print('the recall score is:', recall_score(y_test, prediction))
    print('the precision score is:', precision_score(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of svm model is:", metrics.accuracy_score(y_test, prediction) * 100)
    # #

    show_roc("Support Vector Machines", y_test, prediction)

def apply_svm(x_train, y_train, x_test, y_test):

    model = SVC()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print('the recall score is:', recall_score(y_test, prediction))
    print('the precision score is:', precision_score(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of svm model is:", metrics.accuracy_score(y_test, prediction) * 100)
    show_roc("Support Vector Machines", y_test, prediction)


def get_knn_hyper_parameters(x_train, y_train):
    hyperparameters = {'n_neighbors': randint(1, 10),
                       'leaf_size': randint(1, 8),
                       'weights': ['uniform', 'distance'],
                       'metric': ['euclidean', 'cityblock']
                       }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=hyperparameters, random_state=1,
                             n_iter=10, cv=cv, verbose=0,
                             n_jobs=-1)
    clf.fit(x_train, y_train)
    print('The Top Parameters Are:', clf.best_params_)
    print('The Top Parameter Score Is:', clf.best_score_)

    return clf.best_params_


def apply_knn(x_train, y_train, x_test, y_test, hyperparameters):
    top_n_neighbors = hyperparameters['n_neighbors']
    top_leaf_size = hyperparameters['leaf_size']
    top_weights = hyperparameters['weights']
    top_metric = hyperparameters['metric']

    model = KNeighborsClassifier(n_neighbors=top_n_neighbors, leaf_size=top_leaf_size, weights=top_weights,
                                 metric=top_metric)
    model = model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print('The recall score is:', recall_score(y_test, prediction))
    print('The precision score is:', precision_score(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of KNN classifier is:", metrics.accuracy_score(y_test, prediction) * 100)

    show_roc("K-Nearest Neighbour", y_test, prediction)

def apply_knn(x_train, y_train, x_test, y_test):

    model = KNeighborsClassifier()
    model = model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(confusion_matrix(y_test, prediction))
    print('The recall score is:', recall_score(y_test, prediction))
    print('The precision score is:', precision_score(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("Accuracy of KNN classifier is:", metrics.accuracy_score(y_test, prediction) * 100)

    show_roc("K-Nearest Neighbour", y_test, prediction)

