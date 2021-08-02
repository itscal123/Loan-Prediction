import pickle
import streamlit as st
from PIL import Image

def output(X_test, y_test):
    st.header("Random Forest")
    st.subheader("Decision Trees")
    st.write("""
    Decision trees are an extremely intuitive and expressive way of understanding data. Fundamentally, a decision tree is comprised of two constructs (nodes): leaves and roots. 
    We begin our analysis at the root of the entire tree. Given some evaluation in relation to the root, we then branch off to the corresponding leaf. This process continues until
    we hit a leaf node (has no children so cannot continue). In many ways, you could imagine a decision tree as a combination of many if/else statements. In two dimensions,
    we can visualize each branch as a decision boundary. Consider the following image that splits the data for each traversal down the tree. 
    """)
    random_forest = Image.open("Decision_Tree.png")
    st.image(random_forest, caption="Decision Tree on whether one should play tennis")

    st.subheader("Estimating Class Probabilities")
    st.write("""
    Class estimation for an instance starts with traversing the tree to find the leaf node that corresponds with the instance. Then we return the ratio of training instances
    of the class in this node.
    """)

    st.subheader("Training Decision Trees")
    st.write("""
    Scikit-Learn uses the Classification and Regression Tree (CART) algorithm to train decision trees. The algorithm first splits the training set into using a single feature and threshold.
    These parameters are produced by minimizing the cost function.q2
    """)
    st.write("*CART cost function for classification*")
    st.latex(r'''
    J \big(k, t_k\big)= \frac{m_{left}}{m} G_{left}+\frac{m_{right}}{m} G_{right}
    ''')
    st.write("*G is given by the Gini impurity*")
    st.latex(r'''
    G_i = 1 -  \sum_{k=1}^{n} p_{i, k}^2
    ''')
    st.write("*We could also instead use Entropy*")
    st.latex(r'''
    H_i = -\sum_{k=1}^{n} p_{i, k}log_2 \big(p_{i, k}\big)
    ''')
    st.write("""
    The algorithm than recursively applies the same logic to split subsequent subsets. Note that CART is a greedy algorithm since it attempts find the optimal split at the top level without
    any regard for future splits. Consequently, greedy algorithms often produce solutions that are not guaranteed to be optimal. 
    """)
    
    st.subheader("Gini impurity or Entropy?")
    st.write("""
    Gini impurity operates by giving a score from 0 (pure) to 1 for each node. A node is considered "pure" if all training instances it applies to belong to the same class. 
    In contrast, entropy draws from the concept of molecular disorder in thermodynamics. In the context of machine learning, entropy is zero when it contains instances of only one class.
    Truthfully, most of the time, Gini impurity and entropy lead to similar trees. While the former is slightly faster to compute; however, it also tends to isolate the most
    frequent class in its own branch of the tree. In contrast, entropy tends to produce slightly more balanced trees. 
    """)

    st.subheader("Ensemble")
    st.write("""
    At its core, an ensemble is a group of predictors. For example, we could train a group of decision trees, each on a different random subset of the training set. Then we obtain the 
    predictions of all the individual tres and keep the only the most frequent prediction, or "votes". This ensemble method is what is called a Random Forest! However, what is the intuition behind
    the success of ensembles? Essentially, due to the law of large numbers, we can expect that probability of many different decision trees will converge to their true probability and that 
    by inspecting many different trees, we can make appropriate predictions by using trees that are more appropriate for the given instance. Note that this only holds if all trees are perfectly
    independent (errors are uncorrelated).
    """)
    
    st.subheader("Random Forests")
    st.write("""
    As mentioned before, random forests are an ensemble of decision trees. We introduce randomness in two ways. First, each decision tree is trained on a random subset of the training data.
    Second, and perhaps more importantly, when growing trees rather than searching for the very best feature to split the training data, we search for the best feature among a random
    subset of features. This results in more diverse trees which results in an overall better ensemble (trees are less dependent of each other).
    """)

    st.subheader("Performance")
    estimators = st.radio("Select number of estimators", options=[100, 250, 500])
    criterion = st.radio("Select criterion", options=("Gini Impurity", "Entropy"))
    st.write("Using the above parameters, the model achieves the following score on the test set")
    with open("random_forest{}_{}.pkl".format(estimators, "gini" if criterion=="Gini Impurity" else "Entropy"), "rb") as f:
        model = pickle.load(f)
    st.write(model.score(X_test, y_test))

    st.subheader("Strengths and Weaknesses of Random Forests")
    st.write("""
    One of the benefits of Decision Trees and Random Forests are that they require minimal data preparation.
    In fact, in this dataset, feature scaling nor centering were required to achieve high accuracy. Furthermore,
    the intuition and expressiveness of decision trees make them very easy to interpret. Although, this doesn't extend
    as much for random forests. Nonetheless, under the context of ensembling, we can see how the intuition behind
    decision trees can be applied to the power of random forests. In fact, note that changing the criterion or
    the number of estimators has little affect on the overall accuracy. Initially, we may believe that dimensionality
    was low enough to be captured by the different combinations of random forests. However, note that by continuosly
    allowing for the creation of nodes, we are increasing the models degree of freedoms and increasing the risk of overfitting. Extending
    this to the context of the dataset, it is possible that our random forest models may have overfit the data and may 
    not achieve high accuracy on new instances. Thus, the amoung of depth that the tree is allowed to grow is an important
    hyperparameter to regularize the model. Another issue with decision trees are instability to small variations in 
    the training data; however, this is generally reduced in random forests due to ensembling.
    """)

