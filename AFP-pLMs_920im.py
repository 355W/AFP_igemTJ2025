import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

from utility import save_prob_label, create_list_train_test920_imbalance

class igemTJModel(nn.Module):
    def __init__(self):
        super(igemTJModel, self).__init__()
        # Convolutional layer for local feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Enhanced LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc_drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc_drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 2)

    def forward(self, prot, data_length):
        # Convolutional processing
        conv_out = self.conv(prot.permute(0, 2, 1)).permute(0, 2, 1)
        data_length = (data_length // 2)  # Length halved after max pooling
        
        # Process variable-length sequences
        sorted_lengths, sorted_idx = data_length.sort(descending=True)
        conv_out = conv_out[sorted_idx]
        packed = pack_padded_sequence(conv_out, sorted_lengths.cpu(),
                                     batch_first=True, enforce_sorted=True)
        
        # LSTM processing
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention mechanism
        attn_weights = self.attention(unpacked).squeeze(-1)
        mask = torch.arange(unpacked.size(1), device=attn_weights.device)[None, :] < sorted_lengths[:, None]
        attn_weights = attn_weights * mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-10)
        context = torch.bmm(attn_weights.unsqueeze(1), unpacked).squeeze(1)
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.fc_drop1(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = F.relu(out)
        out = self.fc_drop2(out)
        
        return self.fc3(out)

def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature0 = []
    train_y = []

    for data in batch_traindata:
        feature0.append(data[0])
        train_y.append(data[1])
    data_length = [len(data) for data in feature0]
    feature0 = torch.nn.utils.rnn.pad_sequence(feature0, batch_first=True, padding_value=0)
    return feature0, torch.tensor(train_y, dtype=torch.long), torch.tensor(data_length)

class BioinformaticsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __getitem__(self, index):
        label = self.Y[index]
        df = pd.read_csv('midData/ProtTran/' + self.X[index], header=None)
        dat = df.values.astype(float).tolist()
        dat = torch.tensor(dat)
        return dat, label
        
    def __len__(self):
        return len(self.X)

def train():
    X_train = [item.strip() for item in lst_path_train_all]
    Y_train = lst_label_train_all

    train_set = BioinformaticsDataset(X_train, Y_train)
    model = igemTJModel()
    model = model.to(device)
    
    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=16,
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        collate_fn=coll_paddding
    )
    
    best_val_loss = float('inf')
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        epoch_loss_train = 0.0
        nb_train = 0
        
        for data_x1, data_y, data_length in train_loader:
            optimizer.zero_grad()
            y_pred = model(data_x1.to(device), data_length)  # data_length remains on CPU
            loss = loss_func(y_pred, data_y.to(device))
            loss.backward()
            optimizer.step()
            
            epoch_loss_train += loss.item()
            nb_train += 1
            
        epoch_loss_avg = epoch_loss_train / nb_train
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss_avg:.4f}')
        
        if epoch_loss_avg < best_val_loss:
            model_fn = "midData/model/afp_model_920im_9493.pkl"
            os.makedirs(os.path.dirname(model_fn), exist_ok=True)
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            print(f"Saved model with loss: {best_val_loss:.4f}")

def test():
    test_set = BioinformaticsDataset(test_path_all, test_label_all)
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=16, 
        shuffle=False,  # No shuffle needed for test set
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True, 
        collate_fn=coll_paddding
    )
    
    model = igemTJModel().to(device)
    model.load_state_dict(torch.load('midData/model/afp_model_920im_9493.pkl'))
    model.eval()
    
    arr_probs = []
    arr_labels = []
    arr_labels_hyps = []
    
    with torch.no_grad():
        for data_x1, data_y, data_length in test_loader:
            y_pred = model(data_x1.to(device), data_length)  
            probs = F.softmax(y_pred, dim=1)
            arr_probs.extend(probs[:, 1].cpu().numpy())
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            arr_labels.extend(data_y.numpy())
            arr_labels_hyps.extend(preds)
    
    print("\n========================== TEST RESULTS ================================")
    print(f'Accuracy: {metrics.accuracy_score(arr_labels, arr_labels_hyps):.4f}')
    tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
    print(f'Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    print(f'MCC: {metrics.matthews_corrcoef(arr_labels, arr_labels_hyps):.4f}')
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Precision: {precision:.4f}')
    
    auc = metrics.roc_auc_score(arr_labels, arr_probs)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(arr_labels, arr_probs)
    aupr = metrics.auc(recall_curve, precision_curve)
    
    print(f'AUC: {auc:.4f}')
    print(f'AUPR: {aupr:.4f}')
    
    result_dir = os.path.join(os.path.dirname(__file__), 'Result')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(result_dir, f'T5-AFP_TJ_920im_9493{timestamp}.csv')
    save_prob_label(arr_probs, arr_labels, save_path)
    print(f"Results saved to: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("midData/model", exist_ok=True)
    
    lst_path_train_all, lst_label_train_all, test_path_all, test_label_all = create_list_train_test920_imbalance()
    
    train()
    test()
    print('Training and evaluation completed.')