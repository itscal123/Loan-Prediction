import pickle
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def output(X_test, y_test):
    st.header("Neural Networks")
    st.subheader("Artifical Neural Networks")
    st.write("""
    Artifical Neural Networks (ANNs) are a machine learning model that draws inspiration from the networks of biological neurons found in the brain. More specifically, the inspiration draws from
    two ideas: connectionism and activations. The basic idea is that much like in nature, a bunch of small units, or "neurons", can be used in powerful ways granted that there are many of them
    and that they can build off each other. Although, the inspiration draws from nature, the current implementation of neural networks are not completely in line with the way we understand
    the brain, but the general idea is sufficient in this case. The study of ANNS is expansive and is growing very rapidly. In this case of our application, we will use a very simple
    multi-layer perceptron (MLP) to create a network that can predict whether an instance will be able to pay back their loan.
    """)

    st.subheader("Architecture")
    st.write("""
    Often, the difficulty with any neural network is the designing its architecture. Much like random forests, neural networks are extremely prone to overfitting so it is essential to select an optimal
    combination of hyperparameters. Unfortunately, the training of neural networks are generally much longer than other machine learning algorithms, so it can be expensive to search for the best 
    set of hyperparameters for a neural network. Generally, a good idea would be use Grid Search Cross Validation to search for the optimal neural network, but for the sake of this demonstration
    (since the dataset has relatively low dimensionality) we will create an arbitrary architecture and choose hyperparameters that we know will perform reasonably well. 
    """)

    st.write("*Architecture of our model*")
    mlp = Image.open("mlp.jpeg")
    st.image(mlp, caption="Input Layer: 14 nodes; Hidden Layer 1: 64 nodes; Hidden Layer 2: 32 nodes; Output Layer: 1 node")

    st.write("""
    Here we have chosen a four layer model: an input layer, two hidden layers, and an output layer.
    We know from the research of others that deep layers tend to perform better which lead to the decision
    to have two hidden layers as opposed to a single, more wider, hidden layer. The decision to
    use the numbers 64 and 32 for the number of neurons in each layer stems from empirical evidence
    that using numbers that are a power of 2 seem to perform better. Finally, since we are creating 
    a binary classifier, the output layer is a single neuron with the sigmoid activation function (refer to
    logistic regression).
    """)

    st.write("""
    Note that we use the Rectified Linear Unit (ReLU) as our activation function. This is due to the fact
    that empirically, ReLU tends to converge better than sigmoid which is likely due to the latter 
    creating issues during backpropagation because of saturation (e.g., vanishing/exploding gradient problem).
    """)

    st.write("*Rectified Linear Unit*")
    relu = Image.open('relu.png')
    st.image(relu, caption="Note that function is undifferentiable at 0, which empirically is not a problem.")

    st.subheader("Training")
    st.write("""
    There is a lot of theoretical details and abstract concepts that are outside of the scope of this application
    (hundreds of books and research papers are available if you are interested), so I'll cover the basics.
    Our model was compiled on the binary cross entropy loss function. Essentially, this function is the standard
    for any binary classifier. For our training optimizer, I decided to go with RMSProp algorithm (Hinton, 2012)
    which is a modification of AdaGrad (Duchi, et al., 2011). The RMSProp algorithm performs fairly well
    in most settings and is go-to optimizer. Finally, the metric used to evaluate our model is accuracy which is 
    fairly straightfoward since we are creating a binary classifier. The model was trained for 50 epochs, although
    early stopping was employed with a patience of 8. Incidentally, the neural network stopped training
    around 14 epochs. 
    """)

    st.subheader("Performance")
    st.write("With the model finally trained, the evaluation on our test set resulted in a score of")
    model = tf.keras.models.load_model("my_keras_model.h5")
    st.write(model.evaluate(X_test, y_test)[1])

    st.subheader("Strengths and Weaknesses of Neural Networks")
    st.write("""
    Without a doubt, neural networks can be extremely powerful due to their high level of flexibility
    and complexity. As seen above, even with a simple MLP we were able to achieve astounding accuracy. However,
    with high degrees of freedom comes the risk of overfitting, so it is important to train models in a way
    that reduces overfitting (i.e., controlling the number of nodes, width of layers, etc.). Despite this, 
    with a good architecture neural networks can achieve state of the art performance as seen through many
    breakthroughs in the machine learning community in the past decade. Although very powerful, the additional complexity
    comes at the cost of increase in computations and time to train. Even with our simple model, it took 
    longer to train this network than a random forest. Furthermore, neural networks operate best when 
    features are scaled, so it is important to prepare the data before hand (which can often be the most 
    difficult part!). Ultimately, although powerful, the potential of neural networks are not immediately
    seen in this specific dataset, but hopefully this demonstration shows that even on simple problems they can
    compete with other machine learning methods (and far exceed existing methods in more complex problems).
    """)
