import pickle
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def logistic_plot():
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    fig = plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    return fig

def predict(model):
    term = st.text_input("Enter loan term in months (integer)")

def output(X_test, y_test):
    st.header("Logistic Regression")
    st.subheader("Estimating Probabilities")
    st.write("""
    Using Scikit-Learn's Logistic Regression, we train a logistic regression model on the dataset. 
    Intuitively, logistic regression is used to estimate whether an instance belongs to a particular
    class. In this case, if the estimated probability is over 50%, then the specific person is likely to
    pay back the loan, otherwise it predicts that the person won't be able to pay back the loan. This 
    makes it a binary classifier. 
    """)
    
    st.write("*Logistic Regression model estimated probability (vectorized form)*")
    st.latex(r'''
        \hat{p} =  h_{\bold{\theta}}\big(\bold{x}\big) = \sigma\big(\bold{x}^{T}\bold{\theta}\big) 
        ''')
    
    st.write("""
    Logistic Regression computes a weighted sum of input features (plus a bias term) and then outputs
    the logistic of the result.
    """)
    
    st.write("*Logistic function*")
    st.latex(r'''
        \sigma\big(t\big)=\frac{1}{1+exp\big(-t\big)} 
        ''')
    st.pyplot(logistic_plot())
    
    st.write("""
    Once the Logistic Regression model has estimated the pobability function, it can easily 
    make predictions on future instances easily.
    """)
    st.write("*Logistic Regression model prediction*")
    st.latex(r'''
         x =\begin{cases}0\;\;\;\;if & \hat{p}<0.5 \\1\;\;\;\;if & \hat{p}\geq0.5\end{cases}  
        ''')
    
    st.write("""
    Note that the logistic function is less than 0.5 when t is less than 0 and greater when t is
    greater than or equal to 0.
    """)
    
    st.subheader("Training and Cost Function")
    st.write("""
    The objective of traiing is to set the parameters such that the model estimates high 
    probabilities for positive instances and low probabilities for negative instances (e.g.,
    minimizing the cost function for a single instance)
    """)
    st.write("*Cost function of a single training instance*")
    st.latex(r'''
        c\big(\bold{\theta}\big)=\begin{cases}-log\big(\hat{p}\big) \;\;\;\;\;\;\;\;\;\;
        if & y=1 \\-log\big(1-\hat{p}\big)\;\;\;\;if & y=0\end{cases}
        ''')
    
    st.write("""
    The decision to use -log(t) is due to the function growing very large as t approaches 0.
    This implies that the cost will be large if the model estimates a probability close to 0 
    for a positive instance. On the other hand, -log(t) is close to 0 when t is close to 1, so
    the cost will be close to 0 if the estimated probability is close to 0 for a negative instance
    or close to 1 for a positive instance. The cost function over the whole training set can be
    written as the average cost over all instances (log loss).
    """)

    st.write("*Logistic Regression cost function (log loss)*")
    st.latex(r'''
        J\big(\bold{\theta}\big)=-\frac{1}{m}\displaystyle\sum_{i=1}^m [y^ilog\big(\hat{p}^i\big)+
        \big(1-y^i\big)log \big(1-\hat{p}^i\big)]
        ''')

    st.write("""
    Although there is no known closed form solution to minimize the log loss, we can utilize
    Gradient Descent (or any other optimization algorithm) to find a solution. Naturally, this
    requires the log loss to be differentiable for backpropagation.
    """)
    st.write("*Logistic cost function partial derivatives*")
    st.latex(r'''
         \frac{\partial}{\partial\theta_j}J\big(\bold{\theta}\big)=
         \frac{1}{m}\displaystyle\sum_{i=1}^m \Big(\sigma\Big(\bold{\theta}^T\bold{x}^i\Big)
         -y^i\Big)x_j^i
        ''')

    st.subheader("Results and Performance")

    regularizer = st.radio(
        "Select regularization",
        options=["L1", "L2"])

    if regularizer == "L1":
        st.write("""
        L1 regularization, also known as Manhattan Distance, is the sum of the magnitudes of the vectors
        in the weight vector space. In other words, the sum of absolute values of the individual 
        parameters. If we define w as vector of feature weights, then the regularization
        term is equal to:
        """)
        st.latex(r'''
        |\bold{w}|_1
        ''')
        st.write("""
        This generally results in a more sparse solution than with L2 regularization. Note that L1 regularization is
        equivalent to the log-prior term that is maximized by maximum a posteriori (MAP) Bayesian inference with an 
        isotropic Laplace prior. Using L1 regularization we achieve the following score on our test set.
        """)
        with open("logit_l1.pkl", "rb") as f:
            model = pickle.load(f)
        st.write(model.score(X_test, y_test)) 
        st.write("""
        Compared to L2 regularization, this model performs better. An explanation that could explain
        this is that the weights of certain variables may be significantly more correlated with the 
        predicted outcome than others. In other words, the covariance matrix is sparse, which 
        L1 regularization achieves more often.
        """)
    else:
        st.write("""
        L2 regularization, also known as weight decay or Tikhonov regularization, is the most common
        form of weight decay is the sum of Euclidean distances of the vectors in the weight vector space.
        In other words, the sum of squared coefficients. Additionally, the sum is multiplied by a factor
        of 1/2 (usually to make the computation of gradients easier). If we define w as a vector of 
        feature weights, then the regularization term is equal to:
        """)
        st.latex(r'''
         \frac{1}{2}\big(\|\bold{w}\|_2\big)^2  
        ''')
        st.write("""
        This generally results in a more dense model solution than with L1 regularization. Note that L2 
        regularization is equivalent to maximum a posteriori (MAP) Bayesian inference with a Gaussian prior 
        on the weights. Using L2 regularization we achieve the following score on our test set.
        """)
        with open("logit_l2.pkl", "rb") as f:
            model = pickle.load(f)
        st.write(model.score(X_test, y_test))
        st.write("""
        Compared to L1 regularization, this model performs worse. A possible explanation could be that
        certain features may not be as relevant, so a more sparse model would have a better representation
        of the data than a more dense one.
        """)

    st.subheader("Strength and Weaknesses of Logistic Regression")
    st.write("""
    Conceptually, logistic regression is simply training a decision boundary across the training set
    that minimizes the cost function. As a result predictions are based off where an instance lies
    in relation to the decision boundary, then applying the logistic function for a probabilitiy. Under the context of
    the loan prediction dataset, this algorithm works fairly well. Consequently, logistic regression
    is another powerful "quick and dirty" machine learning algorithm that works quite well in many cases.
    However, this expressive power drops significantly as dimensionality of the data increases; although, this is
    not as evident since our dataset is not dimensionally complex (less than 20 after dropping irrelevant features).
    Imagine a much more complex task like computer vision where dimensionality can reach up to thousands of dimensions, 
    almost certainly resulting in logistic regression's failure to converge to an acceptable solution. 
    """)