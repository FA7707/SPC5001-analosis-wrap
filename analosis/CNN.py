import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from sklearn import metrics
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
path = "evaluation"

def load_files(run):
    df = pd.read_csv(f"results/Cleaned_datasets/labels_{run}.csv")
    images = []
    class_ = {'Ring': 0, 'Arc_lens' : 1, 'Multi_lens' : 2, 'Non_lens' : 3}
    for i in df['Filename']:
        file = f"results/images/{run}/" + i
        images.append(np.load(file))
    y = np.array(df.Class.map(class_))
    X_prescaled = np.stack(images)
    return X_prescaled, y

def measuring_noise(X_prescaled):
    os.makedirs(path, exist_ok=True)
    noise = np.std(X_prescaled[:, :10, : 10])
    df_noise = pd.DataFrame()
    df_noise['Noise'] = [noise]
    df_noise.to_csv(os.path.join(path,'noise.csv'), index=False)
    return noise
    
def preprocessing(X_prescaled, noise):
    scaled = X_prescaled / noise
    stretched = np.arcsinh(scaled)
    m = stretched.mean(axis=(1, 2), keepdims=True)
    s = stretched.std(axis=(1, 2), keepdims=True)
    normed = (stretched - m) / s
    X = normed
    return X

def split_data(X, y, frac_train=0.6, frac_val=0.2, seed=343):
    """Shuffle the examples, then split them into training, validation and test sets."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_train, n_val = int(frac_train * n), int(frac_val * n)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]


def tensors(X, y):
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    X_train_t = torch.tensor(X_train[:, None], dtype=torch.float32)   
    X_val_t = torch.tensor(X_val[:, None], dtype=torch.float32)
    X_test_t = torch.tensor(X_test[:, None], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)             
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, y_val

def accuracy(model, X_t, y_t):
    """Fraction of images classified correctly (threshold 0.5)."""
    with torch.no_grad():
        y_pred = (model(X_t).numpy().argmax(axis = 1))
    return np.mean(y_pred == y_t.numpy())

def train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, optimiser, loss_fn, n_epochs, batch_size=32):
    """Mini-batch training on the training and validation tensors passed in.
    Returns lists of the training cost, validation cost, training accuracy and validation accuracy after each epoch."""
    J_train, J_val, acc_train, acc_val = [], [], [], []
    n = len(X_train_t)
    for epoch in range(n_epochs):
        perm = torch.randperm(n)                      # a new random order of the training images each epoch
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]      # the indices of the images in this mini-batch
            X_batch, y_batch = X_train_t[idx], y_train_t[idx]
            k = torch.randint(0,4, (1,)).item()
            X_batch = (torch.rot90(X_batch, k, dims=(2, 3)))
            flip = torch.randint(0,2, (1,)).item()
            if flip == 1:
                X_batch = torch.flip(X_batch, dims=(3,))
            y_hat = model(X_batch)     
            J = loss_fn(y_hat, y_batch)          
            optimiser.zero_grad()
            J.backward()
            optimiser.step()
        with torch.no_grad():                        
            J_train.append(loss_fn(model(X_train_t), y_train_t).item())
            J_val.append(loss_fn(model(X_val_t), y_val_t).item())
        acc_train.append(accuracy(model, X_train_t, y_train_t))
        acc_val.append(accuracy(model, X_val_t, y_val_t))
    return J_train, J_val, acc_train, acc_val

def plot_curves(J_train, J_val, acc_train, acc_val, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(J_train, color=BLUE, lw=2, label="training")
    axs[0].plot(J_val, color=ORANGE, lw=2, label="validation")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("cost $J$"); axs[0].set_ylim(bottom=0); axs[0].legend()
    axs[1].plot(100 * np.array(acc_train), color=BLUE, lw=2, label="training")
    axs[1].plot(100 * np.array(acc_val), color=ORANGE, lw=2, label="validation")
    axs[1].axhline(25, color=GREY, ls=":", lw=1); axs[1].text(len(acc_val) - 1, 30, "guessing", ha="right", va="top", color=GREY)
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("accuracy (%)"); axs[1].set_ylim(20, 102); axs[1].legend()
    fig.suptitle(title); plt.tight_layout()
    plt.savefig(os.path.join(path,'Cost curve.png'), dpi=300)
    return fig

def count_parameters(model):
    """Total number of weights and biases in the model."""
    return sum(p.numel() for p in model.parameters())

def build_model():
    torch.manual_seed(0) # set the random number generator to a known state, so that the convolutional filters are the same each time
    block1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2))
    
    block2 = nn.Sequential(
        nn.Conv2d(8, 16, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2))
    
    block3 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2))
    cnn = nn.Sequential(block1, block2, block3, nn.AdaptiveMaxPool2d(3), nn.Flatten(), nn.Linear(288,32), nn.ReLU(), nn.Linear(32,4))
    return(cnn)
    
def train(cnn, X_train_t, y_train_t, X_val_t, y_val_t, n_epochs = 100, lr = 0.002):
    optimiser = torch.optim.Adam(cnn.parameters(), lr)
    J_train, J_val, acc_train, acc_val = train_model(cnn, X_train_t, y_train_t, X_val_t, y_val_t, optimiser, nn.CrossEntropyLoss(), n_epochs)
    df_values = pd.DataFrame()
    df_values['J_train'] = J_train
    df_values['J_val'] = J_val
    df_values['acc_train'] = acc_train
    df_values['acc_val'] = acc_val
    df_values.to_csv(os.path.join(path,'Model_values.csv'), index=False)
    
    print(f"final training cost {J_train[-1]:.3f},  final validation cost {J_val[-1]:.3f}")
    print(f"final training accuracy {acc_train[-1]:.3f},  final validation accuracy {acc_val[-1]:.3f}")
    torch.save(cnn.state_dict(), os.path.join(path,'Model_weights.pt'))
    return(cnn, J_train, J_val, acc_train, acc_val)

def matrix(X_val_t, y_val, cnn, title):
    with torch.no_grad():
        y_pred = (cnn(X_val_t).cpu().numpy().argmax(axis = 1))
    actual = y_val
    predicted = y_pred
    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    df_matrix_values = pd.DataFrame()
    df_matrix_values['Actual'] = actual
    df_matrix_values['Predicted'] = predicted
    df_matrix_values.to_csv(os.path.join(path,f'Matrix_values_{title}.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(6, 5)) 
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Ring', 'Arc_lens', 'Multi_lens', 'Non_lens'])
    cm_display.plot(ax=ax)
    ax.set_title(f"CNN confusion matrix ({title} set)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(path,f'CNN Confusion Matrix {title}.png'), dpi=300)
    return fig

def load_noise():
    df = pd.read_csv(os.path.join(path,'noise.csv'))
    saved_noise = df['Noise'][0]
    return saved_noise

def load_model():
    return os.path.join(path,'Model_weights.pt')
    

if __name__ == "__main__":
    X_prescaled, y = load_files("Train")
    noise = measuring_noise(X_prescaled)
    X = preprocessing(X_prescaled, noise)
    X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, y_val = tensors(X, y)
    cnn = build_model()
    cnn, J_train, J_val, acc_train, acc_val = train(cnn, X_train_t, y_train_t, X_val_t, y_val_t)
    matrix_fig = matrix(X_val_t, y_val, cnn, title)
    curve_fig = plot_curves(J_train, J_val, acc_train, acc_val, f"CNN ({count_parameters(cnn)} parameters)")
    plt.show()