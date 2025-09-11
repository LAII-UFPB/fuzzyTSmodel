import numpy as np
import matplotlib.pyplot as plt
from fuzzymodel import FuzzyTSModel
import time

# THIS EXAMPLE IS NOT FINISHED YET

# ====================== Parameters ======================
N_LAGS = 3          # number of past values used as input (easy to change!)
N_HORIZON = 1       # prediction horizon (1 = one-step-ahead)
abs_error_threshold = 0.01

# ====================== Dataset creation ======================
t = np.arange(0, 8*np.pi, 0.02)

# build a sine wave with different behaviors in segments
y1 = np.sin(t[:len(t)//2])
y2 = -np.sin(t[len(t)//2:len(t)//2+len(t)//3])
exp_arg = -(t[len(t)//2+len(t)//3:] - t[len(t)//2+len(t)//3])
y3 = -np.exp(exp_arg) * np.sin(t[len(t)//2+len(t)//3:])
y = np.concatenate((y1, y2, y3))

# create lagged inputs dynamically based on N_LAGS
X = []
for lag in range(1, N_LAGS+1):
    X.append(np.roll(y, lag))
X = np.array(X)
X, y = X[:, N_LAGS:], y[N_LAGS:]  # trim first N_LAGS values
X = X.T

# split into train and validation
train_size = int(len(t)*0.5)
Xt, yt = X[:train_size], y[:train_size]
yv = y[train_size:]

plt.plot(t[N_LAGS:train_size+N_LAGS], yt, label="Train")
plt.plot(t[train_size+N_LAGS:], yv, label="Validation")
plt.legend()
plt.title("Train and Validation data")
plt.grid()
plt.show()

# ====================== Fuzzy model creation ======================
input_range = [-1.5, 1.5]
output_range = input_range

input_names = [f"var{i}" for i in range(1, N_LAGS+1)]
model = FuzzyTSModel(
    input_names=input_names,
    output_name="out",
    N=7,
    input_range=input_range,
    output_range=output_range
)

# visualize first variable
var1 = model.var_manager.get('var1')
for term, value in var1.get_values(0.5).items():
    print(term, ':', value)
var1.plot()

# train model
model.fit(Xt, yt)

# pruning parameters
model.rule_manager.prune_weight_threshold = 0.01
model.rule_manager.prune_use_threshold = 0
model.rule_manager.prune_window = 100

# ====================== Online prediction loop ======================
def model_update(model, y_true, y_pred, i):
    """
    Update model with past predictions if error exceeds threshold.
    Checks up to N_LAGS steps back.
    """
    for k in range(1, min(N_LAGS+1, len(y_true))):
        if len(y_true) >= (k + (N_LAGS-1)):
            abs_error = np.abs(y_true[-k] - y_pred[-(k+1)])
            if abs_error > abs_error_threshold:
                # build input window using only true values for correction
                x_error = np.array([y_true[-(k+j)] for j in range(N_LAGS)])
                y_pred_corr = model.predict_and_update(
                    x_error.reshape(1, -1),
                    np.array([y_true[-k]]),
                    abs_error_threshold=abs_error_threshold
                )
                y_pred[-(k+1)] = y_pred_corr[0]
                print(f"Model updated (lag {k}) at step {i}, abs error {abs_error:.4f}")
    return y_pred


def get_input_window(y_true, y_pred, horizon):
    """
    Build input window mixing true and predicted values depending on horizon.
    horizon=1 -> only true values
    horizon=2 -> 1 predicted + N_LAGS-1 true
    horizon=3 -> 2 predicted + N_LAGS-2 true
    """
    past = []
    for k in range(1, N_LAGS+1):
        if k < horizon and len(y_pred) >= k:   # use prediction if true not yet available
            past.append(y_pred[-k])
        elif len(y_true) >= k:                 # otherwise use real data
            past.append(y_true[-k])
        else:                                  # fallback at the very beginning
            past.append(0.0)
    return np.array([past])


# initialize prediction and true lists
y_pred = [0 for _ in range(N_LAGS)]
y_true = []

for i in range(len(yv)):
    if i > 0:
        y_true.append(yv[i-1])  # append the newly arrived real value
    if i < N_LAGS:
        continue

    assert len(y_pred) == len(y_true)  # consistency check

    # update model based on past errors
    y_pred = model_update(model, y_true, y_pred, i)

    # build input window (horizon=1 means one-step-ahead prediction)
    x_new = get_input_window(y_true, y_pred, horizon=N_HORIZON)

    # make new prediction
    y_new_pred = model.predict(x_new)[0]
    y_pred.append(y_new_pred)

    # online plotting
    plt.plot(range(len(y_pred)), y_pred, label="Fuzzy Pred")
    plt.plot(range(len(y_true)), y_true, label="Real")
    plt.title("Online Prediction: Real vs Fuzzy Predicted")
    plt.grid()
    plt.legend()
    plt.pause(0.05)
    plt.clf()
    time.sleep(0.1)

# final plot
plt.plot(range(len(y_pred)), y_pred, label="Fuzzy Pred")
plt.plot(range(len(y_true)), y_true, label="Real")
plt.title("Online Prediction: Real vs Fuzzy Predicted")
plt.grid()
plt.legend()
plt.show()
