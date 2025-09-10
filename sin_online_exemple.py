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
x1v, x2v, x3v, yv = x1[train_size:], x2[train_size:], x3[train_size:] ,y[train_size:]
Xv = np.vstack((x1v, x2v, x3v))
Xv = Xv.T
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

# let's say that the input validation data is received online in small batches of 10 each t=0.1s 
# and the true output is received after t=0.2s, showing the past true results
# we will predict the output for each small batch received and after receiving the true output 
# check if the absolute error is above a certain threshold
# if so, we will update the model with the new data
# here we simulate this process
y_pred = []
batch_size = 10
for i in range(0, len(Xv), batch_size):
    X_batch = Xv[i:i+batch_size]
    y_batch = yv[i:i+batch_size]
    
    # predict the output for the current batch
    y_batch_pred = model.predict(X_batch)
    y_pred.extend(y_batch_pred)

    plt.plot(range(0, i+len(y_batch_pred)), y_pred, label="Fuzzy Pred")
    plt.plot(range(0, i+len(y_batch)), yv[0:i+batch_size], label="Real")
    plt.legend()
    plt.title("Online Prediction")
    plt.grid()
    plt.pause(0.1)  # pause to simulate real-time plotting
    plt.clf()  # clear the figure for the next batch
    
    # simulate waiting for the true output to be received
    time.sleep(0.1)  # wait for 0.1s (simulating delay)
    
    # after receiving the true output, check the error and update the model if necessary
    if len(y_batch) == len(y_batch_pred):  # ensure we have true values to compare
        model.predict_and_update(X_batch, y_true=y_batch, abs_error_threshold=0.01)
    time.sleep(0.1)  # wait for another 0.1s (simulating delay)


y_pred = np.array(y_pred)
plt.title("Validation data: Real vs Fuzzy Predicted")
plt.plot(yv, label="Real")
plt.plot(y_pred, label="Fuzzy Pred")
plt.grid()
plt.legend()
plt.show()
## Metrics
#results = model.score(y_pred=y_pred, y_true=yv)
#print(f"MAE: {results['MAE']}\nMAPE: {results['MAPE']}\nRMSE: {results['RMSE']}\nR2: {results['R2']}")
