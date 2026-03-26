import numpy as np
import matplotlib.pyplot as plt
x = np.array([

    [0.4, 0.4], # 800 sqft, 2 bed

    [0.6, 0.4], # 1200 sqft, 2 bed

    [0.7, 0.6], # 1400 sqft, 3 bed

    [0.9, 0.6], # 1800 sqft, 3 bed

    [1.1, 0.8], # 2200 sqft, 4 bed

    [1.3, 0.8], # 2600 sqft, 4 bed

    [1.5, 1.0], # 3000 sqft, 5 bed

    [1.8, 1.0]  # 3600 sqft, 5 bed

])



# Targets: Price in Lakhs (normalized by 100)

y = np.array([[0.35], [0.50], [0.65], [0.80], [1.00], [1.20], [1.45], [1.75]])

input_size = 2
hidden_layers_1 = 8
hidden_layers_2 = 8
output_size = 1
lr = 0.01
lr1 = 0.01
lr2 = 0.01
lr3 = 0.01

loss=[]

w1 = np.random.randn(input_size,hidden_layers_1)*np.sqrt(2/input_size)
w2 = np.random.randn(hidden_layers_1,hidden_layers_2)*np.sqrt(2/hidden_layers_1)
w3 = np.random.randn(hidden_layers_2,output_size)*np.sqrt(2/hidden_layers_2)
b1 = np.zeros((1,hidden_layers_1))
b2 = np.zeros((1,hidden_layers_2))
b3 = np.zeros((1,output_size))

def relu(x):
   return np.where(x>0,x,0.01*x)
def relu_derivative(x):
   return np.where(x>0,1,0.01)

for nums in range(1,10001):
    hidden_input = x@w1 + b1
    hidden_output_1 = relu(hidden_input)
    hidden_output_2 = relu(hidden_output_1@w2 + b2)
    y_pred =(hidden_output_2@w3 + b3)

    output_error = (y_pred - y)
    d_output_error = output_error

    hidden_error_2 = d_output_error@w3.T
    d_hidden_error_2 = hidden_error_2*relu_derivative(hidden_output_1@w2 + b2)

    hidden_error_1 = d_hidden_error_2@w2.T
    d_hidden_error_1 = hidden_error_1*relu_derivative(hidden_input)
    w1_prev = w1
    w1-=(x.T@d_hidden_error_1)*lr1
    w1f = (np.square(np.mean(w1_prev) - np.mean(w1)))**0.5
    if w1f < 0.0001:
        lr1+=0.0001
    else:
        lr1*=0.95
    w2_prev = w2
    w2-=(hidden_output_1.T@d_hidden_error_2)*lr2
    w2f = (np.square(np.mean(w2_prev) - np.mean(w2)))**0.5
    if w2f < 0.0001:
        lr2+=0.0001
    else:
        lr2*=0.95
    w3_prev = w3
    w3-=(hidden_output_2.T@d_output_error)*lr3
    w3f = (np.square(np.mean(w3_prev) - np.mean(w3)))**0.5
    if w3f < 0.0001:
        lr3+=0.0001
    else:
        lr3*=0.95
    b1-=(np.sum(d_hidden_error_1 , axis = 0 , keepdims = True))*lr
    b2-=(np.sum(d_hidden_error_2 , axis = 0 , keepdims = True))*lr
    b3-=(np.sum(d_output_error , axis = 0 , keepdims = True))*lr
    if nums%1000 == 0:
        l = np.mean(np.square(y_pred - y))
        loss.append(l)
plt.plot(loss)
plt.title('Training Loss Over Time')
plt.xlabel('Iterations (x1000)')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
print("This is a house price predictor based square feet and number of bedrooms.")
x1 = int(input("enter square feet:"))/2000
x2 = int(input("enter number of bedrooms:"))/5

z = np.array([[x1,x2]])
value = (relu(relu(z@w1 + b1)@w2 + b2)@w3 + b3)*100
print(f"The price of the house is {value[0][0]:.2f} lakhs")


