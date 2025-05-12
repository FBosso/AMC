#import section
import h5py
import torch
import numpy as np
from dataset import RadioDataset
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy
from sklearn.manifold import TSNE
from matplotlib.colors import to_rgba
import os
import matplotlib.pyplot as plt
from datetime import datetime
from models import SelfAttnModel, StandardModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def return_data_train():
    
    # Load data
    train_data = h5py.File('HKDD_AMC12/HKDD_AMC12_train.mat')
    data_raw = torch.tensor(train_data['XTrainIQ'][:], dtype=torch.float32)  # shape: (N, 1, 128, 2)
    data_feature = torch.tensor(train_data['Feature'][:], dtype=torch.float32)  # shape: (N, 1, 228)

    label_base = np.arange(0, 12)
    label_train = label_base.repeat(1000)
    label_train = np.tile(label_train, 21)
    n_classes = 12
    label_train_oh = torch.tensor(np.eye(n_classes)[label_train], dtype=torch.float32)
    
    return data_raw, data_feature, label_train_oh


def return_data_val():
    # Load data
    train_data = h5py.File('HKDD_AMC12/HKDD_AMC12_test.mat')
    data_raw = torch.tensor(train_data['XTrainIQ'][:], dtype=torch.float32)  # shape: (N, 1, 128, 2)
    data_feature = torch.tensor(train_data['Feature'][:], dtype=torch.float32)  # shape: (N, 1, 228)

    # Create labels
    label_base = np.arange(0, 12)
    label_test = label_base.repeat(500)              # 500 samples per class
    label_test = np.tile(label_test, 21)             # 21 blocks → total 12600 samples
    n_classes = 12
    label_test_oh = torch.tensor(np.eye(n_classes)[label_test], dtype=torch.float32)  # shape: (12600, 12)

    # Sampling: 10 samples every 500 (per chunk)
    chunk_size = 500
    samples_per_chunk = 10
    num_chunks = data_raw.size(0) // chunk_size

    selected_indices = []

    for i in range(num_chunks):
        start = i * chunk_size
        indices = torch.randperm(chunk_size)[:samples_per_chunk] + start
        selected_indices.append(indices)

    selected_indices = torch.cat(selected_indices)

    # Slice all three tensors
    data_raw_sampled = data_raw[selected_indices]
    data_feature_sampled = data_feature[selected_indices]
    label_test_oh_sampled = label_test_oh[selected_indices]
    
    return data_raw_sampled, data_feature_sampled, label_test_oh_sampled
    


def return_data_test():
    # Load data
    train_data = h5py.File('HKDD_AMC12/HKDD_AMC12_test.mat')
    data_raw = torch.tensor(train_data['XTrainIQ'][:], dtype=torch.float32) #[125500:,:,:,:]  # shape: (N, 1, 128, 2)
    data_feature = torch.tensor(train_data['Feature'][:], dtype=torch.float32) #[125500:,:]  # shape: (N, 1, 228)

    # python
    label_base = np.arange(0, 12)
    label_test = label_base.repeat(500)
    label_test = np.tile(label_test, 21)  # the class label of test set
    # modulation_name = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK', '8FSK', '16QAM', '32QAM', '64QAM', '4PAM', '8PAM']
    # all_dB = list(np.arange(-20, 21, 2))
    n_classes = 12
    label_test_oh = torch.tensor(np.eye(n_classes)[label_test], dtype=torch.float32)#[125500:,:]
    
    return data_raw, data_feature, label_test_oh


def plot_training_loss(training_loss, validation_loss, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_over_chunks(model, data_raw, data_feature, labels_oh, chunk_size, device, save_path_fig, save_path_data):
    model.eval()
    correct = []
    with torch.no_grad():
        for i in range(0, len(data_raw), chunk_size):
            X1 = data_feature[i:i+chunk_size].to(device)
            X2 = data_raw[i:i+chunk_size].to(device)
            #transform one hot encoding into index
            labels = torch.argmax(labels_oh[i:i+chunk_size], dim=1).to(device)
            #do inference
            preds = model(X1, X2)
            #instantiate softmax
            m = nn.Softmax(dim=1)
            #compute probs
            y_probs = m(preds)
            #intantiate accuracy
            acc_fun = Accuracy(task="multiclass", num_classes=y_probs.shape[1]).to(device)
            #compute accuracy
            acc = acc_fun(y_probs, labels)
            #append values to list
            correct.append(acc.to("cpu"))
    #save list
    array_results = np.array(correct)
    np.save(save_path_data, array_results)        

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(-20,21,2), correct, marker='o')
    plt.xlabel(f"SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_fig)
    plt.close()

def save_experiment_outputs(model, training_loss, validation_loss,  data_raw, data_feature, labels_oh, device, experiment_name="experiment", chunk_size=500):
    # Create timestamped subfolder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{experiment_name}_{timestamp}"
    save_dir = os.path.join("models", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save training loss plot
    loss_plot_path = os.path.join(save_dir, "training_loss.png")
    plot_training_loss(training_loss, validation_loss, loss_plot_path)
    print(f"Training loss plot saved to {loss_plot_path}")

    # Save accuracy plot
    accuracy_plot_path = os.path.join(save_dir, "accuracy_chunks.png")
    data_plot_path = os.path.join(save_dir, "accuracy_chunks_data.npy")
    plot_accuracy_over_chunks(model,  data_raw, data_feature, labels_oh, chunk_size, device, accuracy_plot_path, data_plot_path)
    print(f"Accuracy plot saved to {accuracy_plot_path}")




def plot_accuracy_over_chunks_2(model, data_raw, data_feature, labels_oh, chunk_size, device):
    model.eval()
    correct = []
    with torch.no_grad():
        for i in range(0, len(data_raw), chunk_size):
            X1 = data_feature[i:i+chunk_size].to(device)
            X2 = data_raw[i:i+chunk_size].to(device)
            #transform one hot encoding into index
            labels = torch.argmax(labels_oh[i:i+chunk_size], dim=1).to(device)
            #do inference
            preds = model(X1, X2)
            #instantiate softmax
            m = nn.Softmax(dim=1)
            #compute probs
            y_probs = m(preds)
            #intantiate accuracy
            acc_fun = Accuracy(task="multiclass", num_classes=y_probs.shape[1]).to(device)
            #compute accuracy
            acc = acc_fun(y_probs, labels)
            #append values to list
            correct.append(acc.to("cpu"))
            

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(-20,21,2), correct, marker='o')
    plt.xlabel(f"SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def accuracy_comparison(path_experiment_1, path_experiment_2):
    name_1 = path_experiment_1.split("_")[0]
    data_1 = np.load(f"{path_experiment_1}/accuracy_chunks_data.npy")
    name_2 = path_experiment_2.split("_")[0]
    data_2 = np.load(f"{path_experiment_2}/accuracy_chunks_data.npy")

    
    plt.figure(figsize=(8, 5))
    x = np.arange(-20,21,2)
    plt.plot(x,data_1, marker='o', label=name_1)
    plt.plot(x,data_2, marker='*', label=name_2)
    plt.xlabel(f"SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    
    
def plot_tsne_with_snr_balanced(input1: torch.Tensor, input2: torch.Tensor, labels: torch.Tensor, 
                                 snr_step=6000, samples_per_bin=100):
    """
    Generates balanced t-SNE plots from a subsample of input1 and input2, shaded by SNR.

    Parameters:
    - input1: Tensor of shape (N, 1, 512, 2)
    - input2: Tensor of shape (N, 228)
    - labels: Tensor of shape (N, 12), one-hot encoded
    - snr_step: Number of samples per SNR step (default 6000)
    - samples_per_bin: Number of samples to draw per (class, SNR) bin
    """
    class_names = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                   '8FSK', '16QAM', '32QAM', '64QAM', '4PAM', '8PAM']

    N = input1.shape[0]
    assert input1.shape[0] == input2.shape[0] == labels.shape[0], "Input shapes mismatch"

    # Prepare data
    labels_np = torch.argmax(labels, dim=1).cpu().numpy()
    input1_flat = input1.view(N, -1).cpu().numpy()
    input2_flat = input2.cpu().numpy()

    # Compute SNR bins
    total_snr_bins = N // snr_step
    snr_levels = (np.arange(N) // snr_step)
    snr_norm = (snr_levels - snr_levels.min()) / (snr_levels.max() - snr_levels.min() + 1e-8)

    # Balanced sampling: collect indices for each (class, snr_bin) pair
    selected_indices = []
    for snr_bin in range(total_snr_bins):
        for cls in range(12):
            bin_mask = (snr_levels == snr_bin) & (labels_np == cls)
            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) >= samples_per_bin:
                chosen = np.random.choice(bin_indices, samples_per_bin, replace=False)
            else:
                chosen = np.random.choice(bin_indices, min(len(bin_indices), samples_per_bin), replace=True)
            selected_indices.extend(chosen)
    
    # Subset data
    selected_indices = np.array(selected_indices)
    x1 = input1_flat[selected_indices]
    x2 = input2_flat[selected_indices]
    y = labels_np[selected_indices]
    
    # Enhance SNR-based alpha shading
    snr_raw = snr_norm[selected_indices]
    alpha_power = 2  # Emphasize opacity contrast
    snr_selected = snr_raw ** alpha_power

    # Define colors
    num_classes = labels.shape[1]
    base_colors = plt.cm.get_cmap('tab10', num_classes)

    def compute_tsne_and_plot(data, title):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(data)

        plt.figure(figsize=(10, 8))
        for class_idx in range(num_classes):
            indices = y == class_idx
            rgba = np.array([to_rgba(base_colors(class_idx), alpha) for alpha in snr_selected[indices]])
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                        c=rgba, s=10)

            # Dummy scatter for legend with full opacity
            plt.scatter([], [], color=base_colors(class_idx), label=class_names[class_idx])

        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    compute_tsne_and_plot(x1, "Balanced t-SNE of Input1 (1x512x2)")
    compute_tsne_and_plot(x2, "Balanced t-SNE of Input2 (228)")
    plt.show()


    


def plot_snr_grouped_confusion_matrices(y_true, y_pred):
    """
    Plots 4 lightweight confusion matrices grouped by SNR ranges: (-20,-10), (-10,0), (0,10), (10,20)

    Parameters:
    - y_true: array-like of shape (126000,), ground truth class indices (0-11)
    - y_pred: array-like of shape (126000,), predicted class indices (0-11)
    """
    class_names = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                   '8FSK', '16QAM', '32QAM', '64QAM', '4PAM', '8PAM']

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Generate SNR per sample
    snr_levels = np.arange(-20, 22, 2)  # [-20, -18, ..., 20], 21 steps
    snr_per_sample = np.repeat(snr_levels, 6000)  # 126000 entries

    snr_ranges = [(-20, -10), (-10, 0), (0, 10), (10, 20)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for idx, (low, high) in enumerate(snr_ranges):
        mask = (snr_per_sample >= low) & (snr_per_sample < high)
        y_true_range = y_true[mask]
        y_pred_range = y_pred[mask]

        cm = confusion_matrix(y_true_range, y_pred_range, labels=np.arange(12), normalize='true')

        ax = axs[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"SNR ∈ [{low}, {high}) dB", fontsize=12)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7, color='black')

    plt.tight_layout()
    fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.015, pad=0.02)
    plt.show()


def plot_attention_maps(data_features, data_raw):
    
    model_standard = StandardModel(228,12)
    attention_standard = model_standard(data_features,data_raw,return_attention=True)
    attention_standard = np.array(attention_standard.to("cpu").detach())
    plt.imshow(attention_standard)
    plt.show()
    
    model_self = SelfAttnModel(228,12)
    attention_self = model_self(data_features,data_raw,return_attention=True)
    attention_self = np.array(attention_self.to("cpu").detach())
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_self)
    plt.axis('off')
    plt.show()