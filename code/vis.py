import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


model_data = np.load('../ckpt/best_model_lr1_reg0_relu_hidden1024_512.npz')
W1 = model_data['W1'] 


def visualize_all_filters(W1, num_cols=32):
    H = W1.shape[1]
    num_rows = int(np.ceil(H / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i in range(H):
        row, col = divmod(i, num_cols)
        w = W1[:, i]
        w_img = w.reshape(3, 32, 32).transpose(1, 2, 0)
        w_img = (w_img - w_img.min()) / (w_img.max() - w_img.min() + 1e-5) 
        
        if num_rows == 1:
            ax = axes[col]
        elif num_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]

        ax.imshow(w_img)
        ax.axis('off')
    

    for j in range(H, num_rows * num_cols):
        row, col = divmod(j, num_cols)
        if num_rows == 1:
            axes[col].axis('off')
        elif num_cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.suptitle('First Layer Weights (W1) Visualization', fontsize=14)
    plt.subplots_adjust(top=0.92)
    plt.savefig('./vis.png',dpi=300)
    
visualize_all_filters(W1)
