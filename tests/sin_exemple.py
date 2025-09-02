import numpy as np
import matplotlib.pyplot as plt
from fuzzyTSmodel.fuzzymodel import FuzzyTSModel

# Dataset creation: Using a sine wave with 3 shifts to the left as input variables
# and the original sine wave as output variable
# The model will learn to predict the sine wave based on its past values
# (time series forecasting)

t = np.arange(0,4*np.pi,0.02)
y = np.sin(t)
# 3 shifts to the left
x1 = np.roll(y, 1)
x2 = np.roll(y, 2)
x3 = np.roll(y, 3)
x1, x2, x3,  y = x1[3:], x2[3:], x3[3:], y[3:]

# division in train and validation data:
x1t, x2t, x3t, yt = x1[:int(len(t)*0.7)], x2[:int(len(t)*0.7)], x3[:int(len(t)*0.7)], y[:int(len(t)*0.7)]
Xt = np.vstack((x1t, x2t, x3t))
Xt = Xt.T
x1v, x2v, x3v, yv = x1[int(len(t)*0.7):], x2[int(len(t)*0.7):], x3[int(len(t)*0.7):] ,y[int(len(t)*0.7):]
Xv = np.vstack((x1v, x2v, x3v))
Xv = Xv.T


# variable ranges
input_range = [-1.5, 1.5]
output_range = input_range

# create the fuzzy model
model = FuzzyTSModel(input_names=["var1", "var2", "var3"], output_name= "out", num_regions=7,
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

## model prediction
y_pred = model.predict(Xv)

plt.plot(yv, label="Real")
plt.plot(y_pred, label="Fuzzy Pred")
plt.legend()
plt.show()

# Learned rules
print(model.explain())
