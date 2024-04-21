import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from sklearn.svm import SVC



st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Face Recognition Model on LFW Dataset")

st.write("# Using HOG + PCA + SVM Model")



st.write("Loading data with minimum faces per person as 40 ...")
# Loading and preprocessing the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.4)
X = lfw_people.images
y = lfw_people.target
st.write(f"Loaded all the images where each image is of size : {X.shape[1]}*{X.shape[2]}")

# Extracting HOG features
hog_features = []
for image in X:
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(4,4),
                               cells_per_block=(2,2), visualize=True)
    hog_features.append(features)

st.write("Extracting HOG Features ...")

# Splitting
X_hog = np.array(hog_features)
y = lfw_people.target
X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.15, random_state=42)
st.write(f"HOG Features extracted : {X_train.shape[1]}")





class PCA_class:
  def __init__(self,n_components=None):
    # To initialize a PCA object

    # Total number of reduced features in projected data
    self.n_components = n_components
    self.eigenvalues = None
    self.eigenvectors=None
    self.means=None

  def fit(self,X):
    # To fit the model and finding principal components given an array X.
    if self.n_components==None:
      self.n_components=X.shape[0]
    self.means = np.mean(X,axis=0)
    X_mean = X-self.means
    cov_matrix = (X_mean.T @ X_mean)/X_mean.shape[0]
    cov_matrix = (cov_matrix + cov_matrix.T)/2
    self.eigenvalues, self.eigenvectors = np.linalg.eig(cov_matrix)
    self.eigenvalues = np.abs(np.real(self.eigenvalues))
    self.eigenvectors = np.real(self.eigenvectors)

    # (v) Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(self.eigenvalues)[::-1]
    self.eigenvalues = self.eigenvalues[sorted_indices]
    self.eigenvectors = self.eigenvectors[:, sorted_indices]
    self.eigenvectors = self.eigenvectors/np.sqrt((np.sum(self.eigenvectors*self.eigenvectors, axis=0)))
    return X_mean @ self.eigenvectors[:,:self.n_components]

  def fit_transform(self,X):
    # returns the given array after projected it along principal components
    return (X-self.means) @ self.eigenvectors[:,:self.n_components]


  def explained_variance(self):
    # returns first 'n_components' number of eigen values in decreasing order
    return self.eigenvalues[:self.n_components]
  def explained_variance_ratio(self):
    # returns ratio of variance captured by first 'n_components' number of eigen values in decreasing order
    return self.eigenvalues[:self.n_components]/np.sum(self.eigenvalues)

  def components(self):
    # returns principal components
    return self.eigenvectors[:,:self.n_components]

  def get_eigenvalues(self):
    # returns all eigen values
    return self.eigenvalues
  def get_eigenvectors(self):
    # returns all eigenvectors
    return self.eigenvectors
  




num_components=600   # To reduce the data into 200 features.

pca = PCA_class(n_components=num_components)
X_projected = pca.fit(X_train)
X_test_projected = pca.fit_transform(X_test)
principal_components = pca.components()
# Convert principal components back to eigen faces.
# eigenfaces = (principal_components.T).reshape((principal_components.shape[1],50,37))

# Reconstruct the original data from the extracted features.
# X_reconstructed = (X_projected @ (principal_components.T)) + np.mean(X_train,axis=0)

# To plot the graph of total variance captured by first 'num_components' eigen vectors.
x = np.arange(1,X_train.shape[1]+1,1)
y = np.cumsum(pca.get_eigenvalues())/np.sum(pca.get_eigenvalues())
plt.figure(figsize=(10,7))
plt.plot(x,y)

x = [num_components,num_components]
y = [0.25,1]
plt.plot(x,y,'r')
plt.title("Fraction of total variance captured by first 'num_components' features.",size=20)
plt.xlabel("Num_components",size=15)
plt.ylabel("Variance Ratio Captured",size=15)
st.pyplot()




## Linear Kernel SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
st.write(f"Linear Kernel Accuracy : {accuracy_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
st.pyplot()


# Polynomial Kernel SVM
param_grid = {
              'degree' : [2,3,4],
              'gamma': np.logspace(-2,2,4)}
svm = SVC(kernel='poly')
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
st.write(f"Best parameters for Poly Kernel : {best_params}")
y_pred = grid_search.best_estimator_.predict(X_test)
st.write(f"Polynomial Kernel Accuracy : {accuracy_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
st.pyplot()


# RBF Kernel SVM
param_grid = {
              'C': np.logspace(-1,3,5),
              'gamma': np.logspace(-3,1,5)}
svm = SVC(kernel='rbf')
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
st.write(f"Best parameters for RBF Kernel : {best_params}")
y_pred = grid_search.best_estimator_.predict(X_test)
st.write(f"RBF Kernel Accuracy : {accuracy_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
st.pyplot()