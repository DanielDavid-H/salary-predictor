\# **PROJECT REPORT**



\# Introduction



This project aims to show the power and robustness of MLP(Multi Layer Perceptrons). We use a method called Stochastic Gradient descent with a custom Adam optimiser to predict the price of houses.

we also will like to show the robustness of deep neural networks by using a very small dataset and using numpy only.



\# Concepts being used



For this project we use a input layer, 2 hidden layers and a output layer. We only the leaky relu function as the activation function as this is a regression problem, using leaky relu will help the model to converge way faster and unlike the regular relu function it doesn't let the neurons die permanently by giving them a very small value so they can wake up again.



The formula for leaky relu is:

&#x20;              

&#x20;            f(x) = {x>0,x



&#x20;                    x<0,0.01\*x}



&#x20;            f'(x) = {x>0,1



&#x20;                    x<0,0.01}



Lastly we use a custom Adam optimiser to increase the learning rate if the weights are updated by a negligible amount and decrease the learning if the weights are bouncing all over the place



\#Working principle



The input is a array of dimensions(8x2) and the output is a array of dimensions(2x8) and the hidden layers have 8 neurons each. For the Adam optimiser to work we give each weights their own learning rates and a constant learning rate for the biases and the weights and biases are initialized using HE initialization to prevent the vanishing gradient problem. This will ensure that the model is not hyper sensitive. We define the weights and biases as numpy matrices. After we define the leaky relu function, we train the model over 10000 iterations. By using partial derivations we are able to find the magnitude and direction of the error of the layer and by using the chain rule we are able to pass on the error to the next layers. The custom Adam optimiser works by checking if the weights are being changed in a meaningful magnitude and adjusting the learning rate accordingly , thus helping the model escape a local minima and find the global minima. The model also shows the loss plot which consists of the loss every 1000 iterations. The model finally gets user input for square feet area and number of bedrooms and predicts the house price.



\# Key observations



* From this project we can see that how powerful a deep MLP is. we can see that even with a very small dataset we were able to predict house prices with extremely high levels of accuracy 

&#x20;

* The custom Adam optimiser also played a hude role in the converging of the graph. 

&#x20;

&#x20;   You can see the graphs below:



| Standard SGD | Adaptive Learning Rate |

| :---: | :---: |

| !\[Standard Loss](loss\_without\_adam.png) | !\[Adaptive Loss](loss\_with\_adam.png)



we can see that the optimiser helped the graph to converge musch faster and we can also see that is graph is overall smoother.



\# Implementation details



* Using of HE initialization to prevent the vanishing gradient problem which prevalent when we use leaky relu activation function .
* normalization of data - divided area by 2000 and no. bedrooms by 5.



\# Conclusion 

&#x20;By this project we are able learn about deep MLPs and the math behind them and understanding how they work how work under the hood.





&#x20;







