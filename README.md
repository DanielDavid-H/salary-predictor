# Introduction
This is a house price predictor based on a stochastic gradient descent(SGD) multi layer perceptron neural network with a custom Adam optimiser. This project aims to show the power of MLPs ,that even with a small dataset  you can predict accurately with the given data. 
# Features
* we use a deep neural network with a 4 layer MLP which consists of a input layer, 2 hidden layers and a output layer.

* The data we train on is a small colllection of houses with parameters as square feet and number of bedrooms.

* we intend to use numpy only so we can see the robustness of a MLP.

* The activation function we are going to use is the leaky relu function.
# Architechture
* As this is a regression problem we only use the leaky relu function as the activation function which is written as

f(x) = {x>0,x : x<0,0.01*x}
        
f'(x) = {x>0,1 : x<0,0.01}
* we also use a custom built adam(Adaptive Moment) optimiser to increase or decrease the learning rate of each neuron independently.
* we use MSE(Mean Squared Error) to calculate the error and update the weights and biases.

  MSE = (1/n)*$(predicted value - correct value)**2

# How To setup and run
🚀 How to Run
Follow these steps to set up the environment and run the house price predictor on your local machine.

1. Clone the Repository
Open your terminal and run the following command to download the project:

Bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
2. Install Dependencies
This project requires NumPy for matrix mathematics and Matplotlib for visualizing the training loss.

Bash
pip install numpy matplotlib
(Alternatively, if you have a requirements.txt file: pip install -r requirements.txt)

3. Execute the Model
Run the main script to start the training process.

Bash
python main.py
4. Training & Prediction
Step A: The model will train for 10,000 iterations using backpropagation.

Step B: A Loss Curve window will pop up. Close the window to proceed to the prediction phase.

Step C: Enter the house details when prompted in the terminal:

Square feet: (e.g., 1500)

Number of bedrooms: (e.g., 3)

The model will then output the estimated price in Lakhs.
