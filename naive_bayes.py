import pickle
import streamlit as st


def output(X_test, y_test):
    st.header("Naive Bayes")
    st.write("""
    A simple Bayesian statistics method of constructing classifiers. Despite their namesake, Naive Bayes
    works surprisingly well. THe underyling principle of the technique is simply assuming that the value
    of a particular feature is independent of the value of all other features, given some class. As a result,
    Generally, parameter estimation is performed using maximum likelihood.
    """)

    st.subheader("Probalistic Model")
    st.write("""
    Given an instance, let x be a vector representing some n (independent) features. We can then assign 
    instance probabilities as
    """)
    st.latex(r'''
    p\big(C_k\;|\;x_1,...,x_n\big) 
    ''')
    st.write("for each possible class C.")

    st.write("Using Bayes' Theorem the conditional probability can be decomposed as")
    st.latex(r'''
    p\big(C_k\;|\;\bold{x}\big)=\frac{p\big(C_k\big)p \big(\bold{x}\;|\;C_k \big)}{p \big(\bold{x}\big)} 
    ''')

    st.write("""
    The numerator is only of interest since the denominator does not depend on C. The numerator is equivalent to
    the joint probability model which can written using the chain rule for probability and definition of conditional
    probability as
    """)
    st.latex(r'''
    p\big(C_k,x_1,...,x_n\big)=p \big(x_1\;|\;x_2,...,x_n,C_k\big) p \big(x_2\;|\;x_2,...,x_n,C_k\big) ... p\big(x_{n-1}\;|\;x_n, C_k\big)p \big(x_n\;|\;C_k\big) p \big(C_k\big) 
    ''')

    st.write('If we assume all features are mutally independent (hence "naive") coditional on C we can write the join model as')
    st.latex(r'''
    p \big(C_k\;|\;x_1,...,x_n\big) \displaystyle\propto p \big(C_k\big)  \prod_{i=1}^n p \big(x_i\;|\;C_k\big) 
    ''')

    st.subheader("Classification Model")
    st.write("""
    In order to make the above naive Bayes probability model into a classifer, we must incorporate a decision rule. 
    A common approach is the use maximum a posteriori (MAP) as a decision rule. Under the MAP framework, the classifier
    that assigns a class label to some instance is
    """)
    st.latex(r'''
    \hat{y}=\underset{k \epsilon \big\{1,...,K\big\}}{\arg\max}[p \big(C_k\big)\prod_{i=1}^n p \big(x_i\;|\;C_k\big)]
    ''')

    st.write("In the case of our data, we will be using a Gaussian prior which defines the probabiltiy density of the classifier as")
    st.latex(r'''
    p \big(x=v\;|\;C_k\big) = \frac{1}{ \sqrt{2 \pi  \sigma_k^2 } }e^{- \frac{ \big(v- \mu_k \big)^2 }{2 \sigma _k^2} }
    ''')

    st.subheader("Performance")
    st.write("""
    After training on the train set, the score our Naive Bayes classifier makes on the test set is
    """)
    with open("pickled_models\\naive_bayes.pkl", "rb") as f:
        model = pickle.load(f)
    st.write(model.score(X_test, y_test))

    st.subheader("Stregths and Weaknesses of Naive Bayes")
    st.write("""
    As seen above, the results of the Naive Bayes model was subpar. This largely comes from the fact that
    the independence assumptions that the model makes are quite contrary to the reality of the dataet. However,
    despite this contradiction, the performs suprisingly well. One benefit of the model is the fact that
    it can alleviate the issues of the curse of dimensionality by assuming feature distributions are independent
    from one anothers (i.e., estimating each distribution as a one-dimensional distribution)
    """)