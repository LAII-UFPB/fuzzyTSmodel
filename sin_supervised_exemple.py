import numpy as np
from fuzzymodel import FuzzyTSModel
import matplotlib.pyplot as plt

def build_supervised(y: np.ndarray, n_lags: int, horizon: int):
    """
    Transform a time series into supervised learning format for multi-output forecasting.

    Args:
        y (np.ndarray): 1D time series data
        n_lags (int): number of past values to use as inputs
        horizon (int): number of steps ahead to predict (multi-output size)

    Returns:
        X (np.ndarray): shape (n_samples, n_lags)
        Y (np.ndarray): shape (n_samples, horizon)
    """
    X, Y = [], []
    n_samples = len(y) - n_lags - horizon + 1
    for i in range(n_samples):
        x_i = y[i : i + n_lags]               # past values
        y_i = y[i + n_lags : i + n_lags + horizon]  # next horizon values
        X.append(x_i)
        Y.append(y_i)
    return np.array(X), np.array(Y)


def stream_supervised(y: np.ndarray, n_lags: int, horizon: int):
    """
    Generator for streaming supervised samples from a time series.

    Args:
        y (np.ndarray): 1D time series data
        n_lags (int): number of past values as input
        horizon (int): number of steps ahead as output

    Yields:
        (x_i, y_i): tuple
            x_i -> np.ndarray shape (n_lags,)
            y_i -> np.ndarray shape (horizon,) if available, else None
    """
    for i in range(len(y)):
        if i < n_lags:
            continue  # not enough past data yet
        x_i = y[i-n_lags:i]  # last n_lags values
        if i + horizon <= len(y):
            y_i = y[i:i+horizon]  # next horizon values (ground truth)
        else:
            y_i = None  # not enough future data
        yield np.array(x_i), (np.array(y_i) if y_i is not None else None)



# parameters
# Parameters
N_LAGS = 5
HORIZON = 3
t = np.linspace(0, 20*np.pi, 2000)
y = np.sin(t) + 0.1*np.random.randn(len(t))  # noisy sine

# train data
train_samples = int(0.2*len(y)) # 20% of the total samples
train_data = y[:train_samples]
Xt,yt = build_supervised(train_data, N_LAGS, HORIZON)

# Create generator
stream = stream_supervised(y[train_samples:], N_LAGS, HORIZON)

# Online simulation
model = FuzzyTSModel(
    input_names=[f"lag{i}" for i in range(1, N_LAGS+1)],
    output_name="out",
    N=6,
    input_range=[-2, 2],
    output_range=[-2, 2],
    horizon=HORIZON
)

model.fit(Xt, yt)

y_true = []
y_pred = []
for step, (x_i, y_i) in enumerate(stream):
    if y_i is None:
        break  # no more future ground truth
    # predict
    y_p = model.predict(np.array([x_i]))
    y_pred.extend(y_p)

    
    if step % 100 == 0:
        print(f"Step {step}: x_i={x_i[-3:]}, y_true={y_i}, y_pred={y_p}")

    # online plotting
    plt.plot(range(len(y_pred)), y_pred, label="Fuzzy Pred")
    plt.plot(range(len(y_true)), y_true, label="Real")
    plt.title("Online Prediction: Real vs Fuzzy Predicted")
    plt.grid()
    plt.legend()
    plt.pause(0.05)
    plt.clf()

    # update if needed
    model.predict_and_update(np.array([x_i]), np.array([y_i]), abs_error_threshold=0.1, verbose=False)
    y_true.append(y_i)

