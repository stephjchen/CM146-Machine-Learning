"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
    
        index = 0
        for line in fid:
            for word in extract_words(line): #extract_words return a list of unique words
                if word not in word_list:
                    word_list[word] = index
                    index += 1

        #pass

    # print('word_list length', len(word_list))

    #print word list
    #for word in word_list:
    #   print(word)


        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix

        line_index = 0
        for line in fid:
            words_in_line = extract_words(line)
            
            for word in words_in_line:
                word_index = word_list.get(word)
                feature_matrix[line_index][word_index] = 1 #mark word as present
            
            line_index += 1
                

        #pass
        # print(feature_matrix)
        # print(np.shape(feature_matrix)) #dimension is (630, 1811)
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'       
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    #check metric is in metric list
    metric_list = ["accuracy", "f1_score", "auroc"]
    assert (metric in metric_list)

    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance

    score = -1
    if metric == 'f1_score':
        score = metrics.f1_score(y_true, y_label)
    elif metric == 'accuracy':
        score = metrics.accuracy_score(y_true, y_label)
    elif metric == 'auroc':
        score = metrics.roc_auc_score(y_true, y_label)

    return score
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance

    scores = 0
    #fold = 0
    for train_index, test_index in (kf.split(X, y)): #loop runs 5 times

        X_training, X_testing = X[train_index], X[test_index]
        y_training, y_testing = y[train_index], y[test_index]

        clf.fit(X_training, y_training)
        y_pred = clf.decision_function(X_testing)

        score = performance(y_testing, y_pred, metric)
        scores += score
        #fold += 1
        #print('for fold: ', fold)
        #print('score is: ', score)
        #print('\n')

    #assert(kf.get_n_splits(X, y) == counter) -> correct
    avg_score = scores/kf.get_n_splits(X,y)
    print('avg_score for 5 fold is: ', avg_score)
    print('\n')
    #return average of all scores    
    return avg_score
    ### ========== TODO : END ========== ###


#pass in X_train, y_train
def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation
    
    #TODO: store all scores and sort
    scores = {}
    opt_c = -1
    for c_val in C_range:
        print('testing c: ', c_val)
        clf = SVC(kernel='linear', C=c_val)
        cv_score = cv_performance(clf = clf, X=X, y=y, kf=kf, metric=metric)
        scores[c_val] = cv_score 
    
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    opt_c = sorted_scores[0]        

    return opt_c
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance
    pred = clf.decision_function(X)
    score = performance(y_true=y, y_pred=pred, metric=metric) 

    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc"]
    
    ### ========== TODO : START ========== ###
    # part 1: split data into training (training + cross-validation) and testing set

    X_train = X[0:560]
    y_train = y[0:560]
    X_test = X[560:630]
    y_test = y[560:630]

    #print('X_train', X_train)
    #print('X_train shape', np.shape(X_train))
    #print('\n')
    #print('y_train', y_train)
    #print()
    #print('\n')
    
    # part 2: create stratified folds (5-fold CV)
    kf = StratifiedKFold(n_splits=5)
    
    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV

    for metric in metric_list:
        best_c = select_param_linear(X=X_train, y=y_train, kf=kf, metric=metric)
        print(metric, best_c)

    # part 3: train linear-kernel SVMs with selected hyperparameters
    clf = SVC(kernel='linear', C=10)
    clf.fit(X_train, y_train)

    # part 3: report performance on test data
    for metric in metric_list:
        perf = performance_test(clf, X_test, y_test, metric)
        print(metric, perf)

    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
