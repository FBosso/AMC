#import section
import h5py
import torch
import numpy as np
from dataset import RadioDataset
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy

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



import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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
    plt.plot(np.arange(len(correct)), correct, marker='o')
    plt.xlabel(f"Chunk (each of size {chunk_size})")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Data Chunks")
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
