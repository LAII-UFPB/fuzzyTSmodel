import numpy as np
import matplotlib.pyplot as plt
from fuzzymodel import FuzzyTSModel
import time

# ====================== Dataset creation ======================: 
# Using a sine wave with 3 shifts to the left as input variables
# and the original sine wave as output variable
# The model will learn to predict the sine wave based on its past values
# (time series forecasting)
t = np.arange(0,8*np.pi,0.02)

# the signal is a sine wave with a change in amplitude at the middle of the signal
y1 = np.sin(t[:len(t)//2])
y2 = -np.sin(t[len(t)//2:len(t)//2+len(t)//3])
exp_arg = -(t[len(t)//2+len(t)//3:] - t[len(t)//2+len(t)//3])
y3 = -np.exp(exp_arg)*np.sin(t[len(t)//2+len(t)//3:])  
y = np.concatenate((y1, y2,y3))   

# 3 shifts to the left
x1 = np.roll(y, 1)
x2 = np.roll(y, 2)
x3 = np.roll(y, 3)
x1, x2, x3,  y = x1[3:], x2[3:], x3[3:], y[3:]

# division in train and validation data:
train_size = int(len(t)*0.5)
x1t, x2t, x3t, yt = x1[:train_size], x2[:train_size], x3[:train_size], y[:train_size]
Xt = np.vstack((x1t, x2t, x3t))
Xt = Xt.T
yv = y[train_size:]
plt.plot(t[3:train_size+3], yt, label="Train")
plt.plot(t[train_size+3:], yv, label="Validation")
plt.legend()
plt.title("Train and Validation data")
plt.grid()
plt.show()


# ====================== Fuzzy model creation and usage ======================:

# variable ranges
input_range = [-1.5, 1.5]
output_range = input_range

# create the fuzzy model
model = FuzzyTSModel(input_names=["var1", "var2", "var3"], output_name= "out", N=6,
                          input_range=input_range, output_range=output_range)

# visualizing the created variables 
# here we visualize only the first input variable 
# we are checking the pertinence values for the input 0.5
var1 = model.var_manager.get('var1')
for term, value in var1.get_values(0.5).items():
    print(term, ':',value)

# plot the variable
var1.plot()

# model train
model.fit(Xt, yt)

# pruning parameters
model.rule_manager.prune_weight_threshold = 0.1
model.rule_manager.prune_use_threshold = 2
model.rule_manager.prune_window = 25

# model prediction

# let's say we receive the data at each 0.1s, after 0.3s we will have enough data to make a prediction
# to the 0.4s point, then at 0.5s we will have the true output for the 0.4s point, and so on.
# we will predict the output for each new input data point received
# check if the absolute error is above a certain threshold
# if so, we will update the model with the new data
# here we simulate this process

y_pred = [0 for i in range(3)]  # initial predictions for the first 3 points (not predicted)
y_true = []
abs_error_threshold = 0.01
for i in range(len(yv)):
    if i>0:
        y_true.append(yv[i-1])  # append the true output we are receiving now
    if i<3:
        continue

    # prepare the new input data point
    x1v, x2v, x3v = yv[i-1], yv[i-2], yv[i-3]  # using the true past values as input
    x_new = np.array([[x1v, x2v, x3v]])  # new input data point


    # the prediction and update
    y_new_pred = model.predict_and_update(x_new)[0]  # predict the output for the new input data point
    y_pred.append(y_new_pred)
    
    plt.plot(range(len(y_pred)), y_pred, label="Fuzzy Pred")
    plt.plot(range(len(y_true)), y_true, label="Real")
    plt.title("Online Prediction: Real vs Fuzzy Predicted")
    plt.grid()
    plt.legend()
    plt.pause(0.05)  # pause to update the plot 
    plt.clf()  # clear the plot for the next iteration


    time.sleep(0.1)  # simulate waiting for new data point