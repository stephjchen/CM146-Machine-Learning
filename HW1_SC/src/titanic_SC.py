"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.   
## I will use the "@" symbols to refer to some variables and functions. 
## For example, for the 3 lines of code below
## x = 2
## y = x * 2 
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from matplotlib import pyplot as plt 

######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE. 

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not 
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not

        #training_distr = Counter(y)
        prob1 = sum(y==1)/len(y) #probability of those survived

        prob0 = sum(y==0)/len(y) #prob of those dead
        s_v_d = [prob1, prob0]
        

        self.probabilities_ = s_v_d


        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_

        y = np.random.choice(2, X.shape[0],p=[self.probabilities_[1], self.probabilities_[0] ])    

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set 
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction

    train_error = 0 ## average error over all the @ntrials
    test_error = 0
    train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 

    for i in range(0, ntrials):
    	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = i)
    	clf.fit(X_train, y_train)
    	y_train_pred = clf.predict(X_train)
    	y_test_pred = clf.predict(X_test)
    	train_scores.append(1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True))	
    	test_scores.append(1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True))

    train_error = sum(train_scores)/ntrials
    test_error = sum(test_scores)/ntrials

    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    #TODO: Uncomment this when submitting!!
    # print('Plotting...')
    # for i in range(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    major_clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    major_clf.fit(X, y)                  # fit training data using the classifier
    y_pred = major_clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    random_clf = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    random_clf.fit(X, y)                  # fit training data using the classifier
    y_pred = random_clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    # call the function @DecisionTreeClassifier

    decTree_clf = DecisionTreeClassifier(criterion="entropy") # create DecisionTreeClassifier classifier, which includes all model parameters
    decTree_clf.fit(X, y)                                     # fit training data using the classifier
    y_pred = decTree_clf.predict(X)                           # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """


    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    # call the function @KNeighborsClassifier

    #k=3
    KNN3_clf = KNeighborsClassifier(n_neighbors=3) # create KNeighborsClassifier classifier, which includes all model parameters
    KNN3_clf.fit(X, y)                             # fit training data using the classifier
    y_pred = KNN3_clf.predict(X)                   # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for 3-Nearest Neighbors...: %.3f' % train_error)

    #k=5
    KNN5_clf = KNeighborsClassifier(n_neighbors=5) # create KNeighborsClassifier classifier, which includes all model parameters
    KNN5_clf.fit(X, y)                             # fit training data using the classifier
    y_pred = KNN5_clf.predict(X)                   # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for 5-Nearest Neighbors...: %.3f' % train_error)

    #k=7
    KNN7_clf = KNeighborsClassifier(n_neighbors=7) # create KNeighborsClassifier classifier, which includes all model parameters
    KNN7_clf.fit(X, y)                             # fit training data using the classifier
    y_pred = KNN7_clf.predict(X)                   # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for 7-Nearest Neighbors...: %.3f' % train_error)

    ### ========== TODO : END ========== ###


    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error

    #majority vote classifier
    clfErr = error(major_clf, X, y)
    print('For Majority Vote Classifier:')
    print('\t-- training error: %.3f' % clfErr[0])
    print('\t-- testing error: %.3f' % clfErr[1])
    
    clfErr = error(random_clf, X, y)
    print('For Random Classifier:')
    print('\t-- training error: %.3f' % clfErr[0])
    print('\t-- testing error: %.3f' % clfErr[1])
    
    clfErr = error(decTree_clf, X, y)
    print('For Decision Tree Classifier:')
    print('\t-- training error: %.3f' % clfErr[0])
    print('\t-- testing error: %.3f' % clfErr[1])
    
    clfErr = error(KNN5_clf, X, y)
    print('For 5-Nearest Neighbors Classifier:')
    print('\t-- training error: %.3f' % clfErr[0])
    print('\t-- testing error: %.3f' % clfErr[1])


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))
    cv_score = [] ## track accuracy for each value of $k, should have length 25 once you're done
    errors = []

    for i in k:
        cur_KNN_clf = KNeighborsClassifier(n_neighbors = i)
        acc_scores = cross_val_score(cur_KNN_clf, X, y, cv=10, scoring='accuracy')
        score = np.mean(acc_scores)
        cv_score.append(score)
        #print('\t-- training error for %g-Nearest Neighbors...: %.3f' % (i, 1-score))
        errors.append( (i, 1-score) )

    # plot validation error against each k
    # DONE: comment when done

    # x_val = [x[0] for x in errors]
    # y_val = [x[1] for x in errors]

    # plt.plot(x_val, y_val)
    # plt.scatter(x_val, y_val, s = 5)
    # plt.xlabel('k neighbors')
    # plt.ylabel('Validation Error')

    # plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')

    decTree_errors = []

    for i in range(1,21):
    	dec_i_tree = DecisionTreeClassifier(criterion='entropy',max_depth=i) ## build tree with depth i
    	dec_i_training, dec_i_testing = error(dec_i_tree, X, y)
    	decTree_errors.append( (i, dec_i_training, dec_i_testing) )


    #DONE: plot, comment when done

    # x_val = [x[0] for x in decTree_errors]
    # dec_train_error = [x[1] for x in decTree_errors]
    # dec_test_error = [x[2] for x in decTree_errors]

    # plt.axis(xlim = (0, 21))
    # plt.plot(x_val, dec_train_error, label='training error')
    # plt.scatter(x_val, dec_train_error, s=5)
    # plt.plot(x_val, dec_test_error ,label='test error')
    # plt.scatter(x_val, dec_test_error, s=5)
    # plt.xlabel('tree depth limit')
    # plt.ylabel('Error')
    # plt.legend()

    # plt.show()

    ### ========== TODO : END ========== ###

    #use 90% of X as training data, 10% as testing data
    def error_fraction(clf, X, y, ntrials=100, test_size=0.1, fraction=10):
    #Computes training and test error for a random split of data

        train_error = 0 ## average error over all the @ntrials
        test_error = 0
        train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 
        frac = fraction/10

        for i in range(0, ntrials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = i)
            X_train = X_train[: int(len(X_train)*frac)] #only use fraction of traiing data
            y_train = y_train[: int(len(y_train)*frac)] #only use fraction of training data label

            clf.fit(X_train, y_train)
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            train_scores.append(1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True))  
            test_scores.append(1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True))

        train_error = sum(train_scores)/ntrials
        test_error = sum(test_scores)/ntrials

    ### ========== TODO : END ========== ###

        return train_error, test_error


    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')

    KNN_clf = KNeighborsClassifier(n_neighbors=7) #use k = 7 for KNN
    decTree_clf =  DecisionTreeClassifier(criterion='entropy',max_depth=6)#use depth = 6 for Decision Tree

    decTree_errors = []
    KNN_errors = []

    for i in range(1,11):

        KNN_error = error_fraction(KNN_clf, X, y, 100, 0.1, i)
        decTree_error = error_fraction(decTree_clf, X, y, 100, 0.1, i)

        #used for plotting
        KNN_errors.append( (i, KNN_error[0], KNN_error[1]))
        decTree_errors.append( (i, decTree_error[0], decTree_error[1]) )

        # print('    Using %g%% of training data...' % (i*10))
        # print('\t-- error for decision tree...       --training error: %.3f & --testing error: %.3f' % (decTree_error[0], decTree_error[1]))
        # print('\t-- error for K-Nearest Neighbors... --training error: %.3f & --testing error: %.3f' % (KNN_error[0], KNN_error[1]))

    #plot part h
     #DONE: plot, comment when done

    # x_val = [(x[0]*10) for x in decTree_errors]
    # dec_train_error = [x[1] for x in decTree_errors]
    # dec_test_error = [x[2] for x in decTree_errors]

    # KNN_train_error = [x[1] for x in KNN_errors] 
    # KNN_test_error = [x[2] for x in KNN_errors]

    # plt.plot(x_val, dec_train_error, label='decTree training error')
    # plt.scatter(x_val, dec_train_error, s=5)
    # plt.plot(x_val, dec_test_error ,label='decTree test error')
    # plt.scatter(x_val, dec_test_error, s=5)

    # plt.plot(x_val, KNN_train_error, label='KNN training error')
    # plt.scatter(x_val, KNN_train_error, s=5)
    # plt.plot(x_val, KNN_test_error ,label='KNN test error')
    # plt.scatter(x_val, KNN_test_error, s=5)


    # plt.xlabel('Amount Data Used')
    # plt.ylabel('Error')
    # plt.legend()

    # plt.show()


    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
