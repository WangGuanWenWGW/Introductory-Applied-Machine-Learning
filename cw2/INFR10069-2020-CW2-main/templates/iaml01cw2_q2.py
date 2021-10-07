
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml01cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should define the functions shown below in your code.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission
##########################################################

#--- Code for loading the data set and pre-processing --->
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import copy
import matplotlib.pyplot as plt
from iaml01cw2_helpers import load_FashionMNIST 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#normalisation
sys.path.append('/Users/thatsmilersmiles/Desktop/cw2/INFR10069-2020-CW2-main/helpers')
path = '/Users/thatsmilersmiles/Desktop/cw2/INFR10069-2020-CW2-main/data/fashion'
Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST(path)
Xtrn_orig = copy.deepcopy(Xtrn)
Xtst_orig = copy.deepcopy(Xtst)
Xtrn = Xtrn / 255.0
Xtst = Xtst / 255.0
Xmean = np.mean(Xtrn, axis = 0)
Xtrn_nm = Xtrn - Xmean
Xtst_nm = Xtst - Xmean
Xtrn_nm.shape
#<----

# Q2.1
def iaml01cw2_q2_1():
    lm = LogisticRegression()
    lm.fit(Xtrn_nm, Ytrn)
    Ytst_pred = lm.predict(Xtst_nm)
    accuracy = metrics.accuracy_score(Ytst, Ytst_pred)
    cm = confusion_matrix(Ytst, Ytst_pred)
    print("Accuracy: {0:.2f}%".format(accuracy * 100))
    print("Confusion matrix: \n", cm)
iaml01cw2_q2_1()   # comment this out when you run the function

# Q2.2
def iaml01cw2_q2_2():
    lm = SVC(kernel = 'rbf', C = 1.0, gamma = 'auto')
    lm.fit(Xtrn_nm, Ytrn)
    Ytst_pred = lm.predict(Xtst_nm)
    mean_accuracy = metrics.accuracy_score(Ytst, Ytst_pred)
    cm = confusion_matrix(Ytst, Ytst_pred)
    print("Accuracy: {0:.2f}%".format(mean_accuracy * 100))
    print("Confusion matrix: \n", cm)
iaml01cw2_q2_2()   # comment this out when you run the function

# Q2.3
def iaml01cw2_q2_3():
    pca = PCA(n_components = 2)
    pca.fit(Xtrn_nm)
    components = pca.fit_transform(Xtrn_nm)
    sigma1 = np.std(components[:, 0])
    sigma2 = np.std(components[:, 1])
    lm = LogisticRegression()
    lm.fit(Xtrn_nm, Ytrn)
    x = np.linspace(-5 * sigma1, 5 * sigma1, 100)
    y = np.linspace(-5 * sigma2, 5 * sigma2, 100)
    xv, yv = np.meshgrid(x, y)
    l = list()
    for i in range(len(xv)):
        for j in range(len(yv)):
            coordinates = [xv[i, j], yv[i, j]]
            l.append(coordinates)      
    projected = pca.inverse_transform(l)
    zv = lm.predict(projected).reshape(100, 100)
    plt.subplots(figsize = (10, 8))
    cs = plt.contourf(xv, yv, zv, levels = np.arange(-0.5, 9.5), cmap = plt.cm.get_cmap('coolwarm', 9))
    plt.colorbar(cs, ticks = range(9), label = 'Class')
    plt.clim(-0.5, 8.5)
    plt.xlabel('Standard Deviations For the First Principal Component')
    plt.ylabel('Standard Deviations For the Second Principal Component')
    plt.title('The Decision Regions For Logistic Regression')
    plt.show()
iaml01cw2_q2_3()   # comment this out when you run the function

# Q2.4
def iaml01cw2_q2_4():
    pca = PCA(n_components = 2)
    pca.fit(Xtrn_nm)
    components = pca.fit_transform(Xtrn_nm)
    sigma1 = np.std(components[:, 0])
    sigma2 = np.std(components[:, 1])
    lm = SVC(kernel = 'rbf', C = 1.0, gamma = 'auto')
    lm.fit(Xtrn_nm, Ytrn)
    x = np.linspace(-5 * sigma1, 5 * sigma1, 100)
    y = np.linspace(-5 * sigma2, 5 * sigma2, 100)
    xv, yv = np.meshgrid(x, y)
    l = list()
    for i in range(len(xv)):
        for j in range(len(yv)):
            coordinates = [xv[i, j], yv[i, j]]
            l.append(coordinates)      
    projected = pca.inverse_transform(l)
    zv = lm.predict(projected).reshape(100, 100)
    plt.subplots(figsize = (10, 8))
    cs = plt.contourf(xv, yv, zv, levels = np.arange(-0.5, 10.5), cmap = plt.cm.get_cmap('coolwarm', 10))
    plt.colorbar(cs, ticks = range(10), label = 'Class')
    plt.clim(-0.5, 9.5)
    plt.xlabel('Standard Deviations For the First Principal Component')
    plt.ylabel('Standard Deviations For the Second Principal Component')
    plt.title('The Decision Regions For SVM')
    plt.show()
iaml01cw2_q2_4()   # comment this out when you run the function

# Q2.5
def iaml01cw2_q2_5():
    Xsmall = np.array([])
    for clas in range(10):
        idx = np.where(Ytrn == clas)
        idx1 = idx[0][:1000]
        Xsmall = np.append(Xsmall, Xtrn_nm[idx1])
    Xsmall = Xsmall.reshape(10000, 784)
    Ysmall = np.array([])
    for i in range(10):
        for j in range(1000):
            Ysmall = np.append(Ysmall, i)
    Ysmall = Ysmall.reshape(10000,)
    Cs = np.logspace(-2, 3, num = 10)
    mean_accuracies = []
    for c in Cs:
        lm = SVC(kernel = 'rbf', C = c, gamma = 'auto')
        lm.fit(Xsmall, Ysmall)
        mean_accuracies = np.append(mean_accuracies, np.mean(cross_val_score(lm, Xsmall, Ysmall, cv = 3)))
    plt.plot(Cs, mean_accuracies, 'o-')
    plt.title('Mean Accuracy Against C')
    plt.xlabel('C in Log')
    plt.xscale('log')
    plt.ylabel('Mean CV Classification Accuracy')
    plt.grid()
    plt.show()    
iaml01cw2_q2_5()   # comment this out when you run the function

# Q2.6 
def iaml01cw2_q2_6():
    Cs = np.logspace(-2, 3, num = 10)
    optimalC = Cs[6]
    lm = SVC(kernel = 'rbf', C = optimalC, gamma = 'auto')
    lm.fit(Xtrn_nm, Ytrn)
    Ytrn_pred = lm.predict(Xtrn_nm)
    trn_accuracy =  metrics.accuracy_score(Ytrn, Ytrn_pred)
    print("Training Accuracy: {0:.2f}%".format(trn_accuracy * 100))
    Ytst_pred = lm.predict(Xtst_nm)
    tst_accuracy = metrics.accuracy_score(Ytst, Ytst_pred)
    print("Tesing Accuracy: {0:.2f}%".format(tst_accuracy * 100))
iaml01cw2_q2_6()   # comment this out when you run the function

