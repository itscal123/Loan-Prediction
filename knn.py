import pickle
import streamlit as st
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def pca_plot():  
    angle = np.pi / 5
    stretch = 5
    m = 200

    np.random.seed(3)
    X = np.random.randn(m, 2) / 10
    X = X.dot(np.array([[stretch, 0],[0, 1]])) # stretch
    X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) # rotate

    u1 = np.array([np.cos(angle), np.sin(angle)])
    u2 = np.array([np.cos(angle - 2 * np.pi/6), np.sin(angle - 2 * np.pi/6)])
    u3 = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])

    X_proj1 = X.dot(u1.reshape(-1, 1))
    X_proj2 = X.dot(u2.reshape(-1, 1))
    X_proj3 = X.dot(u3.reshape(-1, 1))

    fig = plt.figure(figsize=(8,4))
    plt.subplot2grid((3,2), (0, 0), rowspan=3)
    plt.plot([-1.4, 1.4], [-1.4*u1[1]/u1[0], 1.4*u1[1]/u1[0]], "k-", linewidth=1)
    plt.plot([-1.4, 1.4], [-1.4*u2[1]/u2[0], 1.4*u2[1]/u2[0]], "k--", linewidth=1)
    plt.plot([-1.4, 1.4], [-1.4*u3[1]/u3[0], 1.4*u3[1]/u3[0]], "k:", linewidth=2)
    plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
    plt.axis([-1.4, 1.4, -1.4, 1.4])
    plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
    plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)

    plt.subplot2grid((3,2), (0, 1))
    plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
    plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)

    plt.subplot2grid((3,2), (1, 1))
    plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
    plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)

    plt.subplot2grid((3,2), (2, 1))
    plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
    plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.axis([-2, 2, -1, 1])
    plt.xlabel("$z_1$", fontsize=18)
    plt.grid(True)
    
    return fig

def output(X_test, X_test_reduced, y_test):
    st.header("K-Nearest Neighbors Classifier")
    st.subheader("K-Neighbors")
    st.write("""
    Simple instance-based learning algorithm that creates a label to designate an instance
    based on frequency among the k (user-defined) neighbors near the query point. A common metric
    is Euclidean space which can be intuitively interpreted. The assumption is that similar things
    exist in close proximity with one another, or "birds of a feather flock together".""")
    
    st.write("""
    As our choice of K decreases to 1, the predictions become less stable. The idea behind this
    is best understood through an example. Consider the following clustering:
    """)

    knn = Image.open("images\knn_image.png")
    st.image(knn, caption="Note that as we decrease k, our classification of the query points changes.")
    
    st.write("""
    The optimal value of k depends on the data. Generally, a larger k reduces the effect of noise
    on the classification. However, this  can result in less distinct boundaries between classes 
    (i.e., overlapping classification boundaries).
    """)
    n_neighbors = st.radio(
        "Select number of neighbors (k)",
        options=[2, 5, 10])
    
    st.subheader("Weight function")
    st.write("""
    There are two main functions for the weighting of neighborhood points. The two functions are as follows:
    """)
    st.write("*Uniform: All points in each neighborhood are weighted equally.*")
    st.write("*Distance: Weighs points by the inverse of their distance. Closer neighbors of a query point will have greater influence than neighbors which are further away.*")
    weights = st.radio(
        "Select weight function used in prediction",
        options=["uniform", "distance"])

    st.subheader("Minowski metric")
    st.write("""
    The Minkowski metric the type of metric we wish to use to measure distance. This parameter is defined as p where
    where p comes in two flavors.
    """)
    st.write("*When p = 1, the algorithm employs Manhattan distance (l1 regularization)*")
    st.write("*When p = 2, the algorithm employs Euclidean distance (l2 regularization)*")
    p = st.radio(
        "Select power parameter for the Minkowski metric",
        options=[1, 2])
    
    st.write("""
    With the selected parameters here is final score evaluated on the test set
    """)
    with open("pickled_models\knn{}_{}_{}.pkl".format(n_neighbors, weights, p), "rb") as f:
        model = pickle.load(f)
    score = model.score(X_test, y_test) 
    st.write(score)
    st.write("""
    Note, that regardless of paramers, KNN performs very poorly. Furthermore, regardless of parameter tuning, 
    all the final scores seem to be roughly the same. These scores highlights the weakness of the KNN
    algorithm: the curse of dimensionality. The good news is that this can be alleviated through the 
    principal component analysis (PCA) technique.
    """)

    pca = st.checkbox("Enable PCA")
    if pca:
        st.subheader("Principal Component Analysis (PCA)")
        st.write("""
        One of the most popular dimensionality reduction algorithms. PCA first identifies the hyperplane that lies closest to the data,
        and then it projects the data onto it.
        """)
        
        st.write("""Note that in order to project the training set onto a lower-dimensional hyperplane, we need to choose the correct hyperplane
        (i.e., the hyperplane that preserves the maximum variance). PCA performs this by finding the an orthogonal hyperplane
        that maximizes variance. This procedure is performed over up to n times which yields the nth axis order
        nth principal component (PC) of the data. We can then project down to some d dimensions.
        """)

        st.write("""
        Consider the following image which visualizes the selection of an orthognal axis or principal
        component that maximizes the variance of the data. 
        """)
        st.pyplot(pca_plot())

        st.write("""
        By incorporating PCA, we can reduce the dimensionality of the training data, which allows KNN to converge to a better
        solution due to the intrinsic reduction is sample space that comes along with dimensionality reduction. In fact, by incorporating
        PCA with the above parameters our final score on the test set increases to
        """)
        
        with open("pickled_models\knn_pca{}_{}_{}.pkl".format(n_neighbors, weights, p), "rb") as f:
            pca_model = pickle.load(f)
        st.write("*Parameters: k = {}, weight = {}, Minkoski metric = {}*".format(n_neighbors, weights, p))
        st.write(pca_model.score(X_test_reduced, y_test))
        st.write("vs")
        st.write(score)

        st.write("""\
        Here we see PCA at work as we significantly increase the predictive power of our model. Additionally, the use of PCA also significantly
        reduced the time it took to train the new models. Furthermore, the use of PCA allowed for the parameters of the KNN algorithm
        to have a more significant impact on the quality of model's predictive power.
        """)
    
    st.subheader("Strengths and Weaknesses of K-Nearest Neighbors")
    st.write("""
    K-NN is a relatively simple algorithm with a lot of power. Furthermore, the intuition behind the algorithm is quite simple
    and easily interpretable. The idea of simply classifying similar points with another makes for a simple implementation
    and a "quick and dirty" solution to our loan prediction problem. However, the algorithm is very prone to the curse
    of dimensionality. In the context of the algorithms, using distance as a metric in higher dimensional spaces is unhelpful since
    the increase in sparsity in higher dimensions results in the increase in distance between query points. 
    """)
