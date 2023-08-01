import streamlit as st
from gen_data import DataGen
from decision_boundary import DecisionBoundary
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# title for the app

st.title('Welcome To the SVM Hyperparameters Tuning and Intuition App')

# sidebar title
st.sidebar.title('Menu')

# select the dtype of the data
dtype = st.sidebar.selectbox(label='Select the type of data to generate',
                             options=['moons','blobs','circles'],
                             index=1)

# generate the data
data = DataGen(dtype=dtype)

# Make X and y
X, y = data.generate_data

# plot the data
st.header('Plot of the Data')
st.pyplot(data.plot_data())

# do the train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

if dtype == 'blobs':
    # take the value of C hyperparameter
    c_val = st.sidebar.number_input(label='Value of C',
                                    min_value=0.01,
                                    max_value=10.0,
                                    value=1.0,
                                    step=0.01,
                                    help='Inverse of Regularization Parameter')

    # button to plot
    btn = st.sidebar.button(label='Plot Decision Boundary')

    if btn:
        # instantiate the clf object
        svm = SVC(kernel='linear', C=c_val)

        # fit on the training data
        svm.fit(X_train,y_train)

        # plot the decision boundary
        boundary = DecisionBoundary(X_train,y_train,clf=svm)
        st.subheader('The decision boundary for the model')

        # plot the decision boundary
        fig_linear = boundary.linear()
        st.pyplot(fig_linear)

        # predict on the test data
        y_pred = svm.predict(X_test)

        # The accuracy score
        score = accuracy_score(y_test,y_pred)
        st.write(f'The accuracy score for the model is = {np.round(score,2)}')

elif dtype == 'moons' or dtype == 'circles':
    # take the value of C hyperparameter
    c_val = st.sidebar.number_input(label='Value of C',
                                    min_value=0.01,
                                    max_value=10.0,
                                    value=1.0,
                                    step=0.01,
                                    help='Inverse of Regularization Parameter')

    # choose the kernel
    kernel = st.sidebar.radio(label='Kernel for Non-Linear data',
                              options=['rbf','poly'],
                              index=0)
    # select the hyperparameter
    if kernel == 'rbf':
        gamma = st.sidebar.number_input(label='Select the Gamma value',
                                        min_value=0.01,
                                        max_value=10.0,
                                        value=1.0,
                                        step=0.01)
    elif kernel == 'poly':
        degree = st.sidebar.number_input(label='Select the degree value for Polynomial',
                                         min_value=2,
                                         max_value=15,
                                         value=2,
                                         step=1)

    mlxtend = st.sidebar.checkbox(label='Plot Boundary using Mlxtend')

    # button to plot
    btn = st.sidebar.button(label='Plot Decision Boundary')

    if btn:
        # instantiate the clf object
        if kernel == 'rbf':
            svm = SVC(kernel=kernel, C=c_val, gamma=gamma)
        elif kernel == 'poly':
            svm = SVC(kernel=kernel,C=c_val, degree=degree)

        # fit on the training data
        svm.fit(X_train, y_train)

        # plot the decision boundary
        boundary = DecisionBoundary(X_train, y_train, clf=svm)
        st.subheader('The decision boundary for the model')


        # plot the decision boundary
        fig_non_linear = boundary.non_linear(mlxtend=mlxtend)
        st.pyplot(fig_non_linear)

        # predict on the test data
        y_pred = svm.predict(X_test)

        # The accuracy score
        score = accuracy_score(y_test, y_pred)
        st.write(f'The accuracy score for the model is = {np.round(score, 2)}')