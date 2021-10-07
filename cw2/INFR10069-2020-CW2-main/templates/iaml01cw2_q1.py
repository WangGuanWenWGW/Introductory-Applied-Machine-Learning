
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml01cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should define the functions shown below in your code.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_my_helpers.py", and import it in this script.
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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error

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

# Q1.1
def iaml01cw2_q1_1():
    print(Xtrn_nm[0,:][0:4])
    print(Xtrn_nm[783,:][0:4])
iaml01cw2_q1_1()   # comment this out when you run the function

# Q1.2
def iaml01cw2_q1_2():
    fig, axs = plt.subplots(10, 5, figsize = (20, 45))
    for clas in range(10):
        dist = (np.sum((Xtrn[Ytrn == clas] - np.mean(Xtrn[Ytrn == clas], 0))**2, axis = 1))**0.5
        clo1 = np.argmin(dist)
        dist_copy1 = copy.deepcopy(dist)
        minimum = np.argmin(dist_copy1)
        dist_copy1[minimum] = np.max(dist)
        clo2 = np.argmin(dist_copy1)
        fur1 = np.argmax(dist)
        dist_copy2 = copy.deepcopy(dist)
        maximum = np.argmax(dist_copy2)
        dist_copy2[maximum] = np.min(dist)
        fur2 = np.argmax(dist_copy2)
        mean = np.mean(Xtrn[Ytrn == clas], axis = 0)
    
        idx = np.argwhere(Ytrn == clas)
        axs[clas, 0].imshow(mean.reshape(28, 28), cmap = 'gray_r') 
        axs[clas, 1].imshow(Xtrn[idx[clo1]].reshape(28, 28), cmap = 'gray_r')
        axs[clas, 2].imshow(Xtrn[idx[clo2]].reshape(28, 28), cmap = 'gray_r')
        axs[clas, 3].imshow(Xtrn[idx[fur2]].reshape(28, 28), cmap = 'gray_r')
        axs[clas, 4].imshow(Xtrn[idx[fur1]].reshape(28, 28), cmap = 'gray_r')
        axs[clas, 0].set_title('Class %s Index = %d' %(clas, idx[clo1]))
        axs[clas, 0].set_xlabel('Mean')
        axs[clas, 1].set_title('Class %s Index = %d' %(clas, idx[clo1]))
        axs[clas, 1].set_xlabel('1st closest')
        axs[clas, 2].set_title('Class %s Index = %d' %(clas, idx[clo2]))
        axs[clas, 2].set_xlabel('2nd closest')
        axs[clas, 3].set_title('Class %s Index = %d' %(clas, idx[fur2]))
        axs[clas, 3].set_xlabel('2nd furthest')
        axs[clas, 4].set_title('Class %s Index = %d' %(clas, idx[fur1]))
        axs[clas, 4].set_xlabel('1st furthest')  
    
iaml01cw2_q1_2()   # comment this out when you run the function

# Q1.3
def iaml01cw2_q1_3():
    pca = PCA(n_components = 5)
    pca.fit_transform(Xtrn_nm)
    return pca.explained_variance_
iaml01cw2_q1_3()   # comment this out when you run the function


# Q1.4
def iaml01cw2_q1_4():
    cm = PCA()
    cm.fit_transform(Xtrn_nm)
    cm.explained_variance_.cumsum()
    variance = cm.explained_variance_ratio_
    cuvar = np.cumsum(cm.explained_variance_ratio_)
    plt.title('Cumulative Explained Variance Ratio')
    lt.xlabel('Principal Component')
    plt.ylabel('Ratio of Variance Explained')
    plt.plot(cuvar, label = 'The cumsum variance ratio')
    plt.legend()
    plt.grid()
    plt.show()
iaml01cw2_q1_4()   # comment this out when you run the function


# Q1.5
def iaml01cw2_q1_5():
    plt.figure(figsize = (10, 5))
    pca = PCA()
    pca.fit(Xtrn_nm)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(pca.components_[i].reshape(28, 28), cmap = 'gray_r')
        plt.axis('off')
        plt.title("Components: " + str(i + 1))
    plt.show()  
iaml01cw2_q1_5()   # comment this out when you run the function


# Q1.6
def iaml01cw2_q1_6():
    K = [5, 20, 50, 200]
    for k in K:
        print("K = ", k)
        pca = PCA(n_components = k, random_state = 1)
        pca.fit(Xtrn_nm)
        for clas in range(10):
            first_index = np.where(Ytrn == clas)[0]
            components = pca.fit_transform(Xtrn_nm)
            reconstructed = pca.inverse_transform(components)
            rmse = np.sqrt(mean_squared_error(Xtrn_nm[first_index[0]], reconstructed[first_index[0]]))
            print("Class {}: {}".format(clas, rmse)) 
iaml01cw2_q1_6()   # comment this out when you run the function


# Q1.7
def iaml01cw2_q1_7():
    fig, axs = plt.subplots(10, 4, figsize = (20, 45))
    K = [5, 20, 50, 200]
    pca = PCA(n_components = K[0])
    for clas in range(10):
        idx = np.where(Ytrn == clas)
        idxx = np.argwhere(Ytrn == clas)
        pca.fit(Xtrn_nm[idx])
        pc = pca.fit_transform(Xtrn_nm[idx])
        reconstructed = pca.inverse_transform(pc[0])
        reconstructed += Xmean
        axs[clas, 0].imshow(reconstructed.reshape(28, 28), cmap = 'gray_r')
        axs[clas, 0].set_title('K = %d Class %d' %(K[0], clas))
    
    pca = PCA(n_components = K[1])
    for clas in range(10):
        idx = np.where(Ytrn == clas)
        idxx = np.argwhere(Ytrn == clas)
        pca.fit(Xtrn_nm[idx])
        pc = pca.fit_transform(Xtrn_nm[idx])
        reconstructed = pca.inverse_transform(pc[0])
        reconstructed += Xmean
        axs[clas, 1].imshow(reconstructed.reshape(28, 28), cmap = 'gray_r')
        axs[clas, 1].imshow(reconstructed.reshape(28, 28), cmap = 'gray_r')
        axs[clas, 1].set_title('K = %d Class %d' %(K[1], clas))
    
    pca = PCA(n_components = K[2])
    for clas in range(10):
        idx = np.where(Ytrn == clas)
        idxx = np.argwhere(Ytrn == clas)
        pca.fit(Xtrn_nm[idx])
        pc = pca.fit_transform(Xtrn_nm[idx])
        reconstructed = pca.inverse_transform(pc[0])
        reconstructed += Xmean
        axs[clas, 2].imshow(reconstructed.reshape(28, 28), cmap = 'gray_r')
        axs[clas, 2].imshow(reconstructed.reshape(28, 28), cmap = 'gray_r')
        axs[clas, 2].set_title('K = %d Class %d' %(K[2], clas))

    pca = PCA(n_components = K[3])
    for clas in range(10):
        idx = np.where(Ytrn == clas)
        idxx = np.argwhere(Ytrn == clas)
        pca.fit(Xtrn_nm[idx])
        pc = pca.fit_transform(Xtrn_nm[idx])
        reconstructed = pca.inverse_transform(pc[0]) 
        reconstructed += Xmean
        axs[clas, 3].imshow(reconstructed.reshape(28, 28), cmap = 'gray_r')
        axs[clas, 3].imshow(reconstructed.reshape(28, 28), cmap = 'gray_r')
        axs[clas, 3].set_title('K = %d Class %d' %(K[3], clas))     

iaml01cw2_q1_7()   # comment this out when you run the function


# Q1.8
def iaml01cw2_q1_8():
    pca = PCA(n_components = 2)
    pca = pca.fit(Xtrn_nm)
    components = pca.fit_transform(Xtrn_nm)
    x = components[:, 0]
    y = components[:, 1]
    plt.subplots(figsize = (15, 12))
    scatter = plt.scatter(x, y, c = Ytrn, s = 3, cmap = plt.cm.get_cmap('coolwarm', 10))
    plt.title('Data in PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    cbar = plt.colorbar(scatter, ticks = range(10), label = 'Color Intensity Referred To Each Class')
    plt.clim(-0.5, 9.5)
    plt.show()
iaml01cw2_q1_8()   # comment this out when you run the function
