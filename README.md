# Streamlit App to visualize the working of Support Vector Classifier on different data types

![alt text](<screenshots/Welcome Screen.png>)

- [Streamlit App to visualize the working of Support Vector Classifier on different data types](#streamlit-app-to-visualize-the-working-of-support-vector-classifier-on-different-data-types)
  - [Commands](#commands)
    - [**Command to run the app**](#command-to-run-the-app)
    - [**To install streamlit in your current virtual env**](#to-install-streamlit-in-your-current-virtual-env)
  - [Introduction](#introduction)
    - [Datasets Supported](#datasets-supported)
  - [Visualizations for Linear Data](#visualizations-for-linear-data)
  - [Visualizations for Non-Linear Data](#visualizations-for-non-linear-data)
    - [Kernel is Polynomial](#kernel-is-polynomial)
    - [Kernel is RBF](#kernel-is-rbf)

## Commands

### **Command to run the app**

```cmd
streamlit run app.py
```

### **To install streamlit in your current virtual env**

```cmd
pip install streamlit
```

## Introduction

This streamlit app is a tool that can be used by students and enthusiasts learning about the functioning of Support Vector Classifier and want to visualize the decision boundary made by the model on **linear** and **non-linear** datasets.

There is also an option to choose between kernels for **non-linear datasets**.

### Datasets Supported

The App supports both **linear** and **non-linear** datasets.

- moons dataset
- concentric circles dataset
- clusters dataset
  
![alt text](<screenshots/Data Types.png>)

## Visualizations for Linear Data

The user can select different values of Hyperparameter `C` that is also the inverse of regularization.

1. Small values of `C` means that the misclassification penalty is low for the models and the margins can maximize their width and smooth out the decision boundary. Low values of `C` avoids the model to overfit and generalize more on the test data thereby reducing model variance.
2. Large values of `C` means that the misclassification penalty is high and the model avoids to misclassify data points as much as possible, reducing the width of margins in the process. Very large values of `C` can potentially lead towards model overfitting.

![alt text](<screenshots/Linear 1.png>)

![alt text](<screenshots/Linear 2.png>)

![alt text](<screenshots/Linear 3.png>)

> **Note- The points that are bold and encircled are the support vector points**

## Visualizations for Non-Linear Data

The optimization function of SVM is:
$$\text{maximize} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(X_i,X_j)$$
where $K(X_i,X_j)$ is the kernel function.

### Kernel is Polynomial

The Polynomial kernel has `degree` as the HyperParameter which tells the amount of polynomial features are used in the kernel function to perform non-linear classification

1. Higher values of `degree` generally leads to overfitting of the model.
2. Lower values of `degree` sometimes is not able to map the training data and might give less accuracy on the training data leading to increased Bias on the training data.

![alt text](<screenshots/Poly 1.png>)

![alt text](<screenshots/Poly 2.png>)

### Kernel is RBF

RBF (Radial Basis Function) kernel is also sometimes called as **universal function approximator** is the best out of the box kernel for Support Vector Classifier.

The HyperParameter used here is `gamma`.

$$\text{rbf} = \large{e^{-\frac{||X_i - X_j||^2}{2\sigma^2}}}$$
or
$$\large{\text{rbf}} = e^{-\gamma||X_i - X_j||^2}$$
where $\gamma = \frac{1}{2\sigma^2}$

**`gamma` is inverse of regularization.**

1. Higher values of `gamma` means that the model will form local boundaries and will stick to the pattern of individual data points leading to the formation of complex decision boundaries resulting in model overfitting.
2. Lower values of `gamma` means that the model will form smooth local decisions leading to more generalized decision boundaries reducing the variance of the model with some tradeoff of the Bias of the model.

![alt text](<screenshots/Gamma 1.png>)

![alt text](<screenshots/Gamma 2.png>)

> **Note- The App also provides an option through the radio button to switch between `matplotlib` plots that are custom built or can switch to `mlxtend` to plot decision boundaries of the model.**

![alt text](<screenshots/custom type.png>)
