import torch
import numpy as np
import pandas as pd
import re
import os
import streamlit as st
from time import sleep
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import datetime

# ======================== Feature Extraction Module ========================
@st.cache_resource
def load_feature_extractor():
    """Load ProtT5 feature extraction model"""
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def extract_prottran_features(sequence, tokenizer, model, device):
    """Extract ProtT5 features for a protein sequence"""
    # Preprocess sequence
    sequence = sequence.replace(" ", "")[:2500]  # Remove spaces and truncate
    sequence = re.sub(r"[UZOB]", "X", sequence)  # Replace uncommon amino acids
    sequence = " ".join(sequence)
    
    # Encode sequence
    ids = tokenizer.batch_encode_plus(
        [sequence], 
        add_special_tokens=True, 
        padding=True,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)
    
    # Extract features
    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Process output
    embeddings = embeddings.last_hidden_state.cpu().numpy()
    seq_len = (attention_mask[0] == 1).sum().item() - 1  # Subtract [CLS] token
    seq_emb = embeddings[0][1:seq_len+1]  # Skip [CLS] token
    
    return seq_emb

# ======================== Prediction Model Module ========================
class igemTJModel(nn.Module):
    def __init__(self):
        super(igemTJModel, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc1 = nn.Linear(256, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc_drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc_drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 2)

    def forward(self, prot, data_length):
        
        conv_out = self.conv(prot.permute(0, 2, 1)).permute(0, 2, 1)
        data_length = (data_length // 2)  
        
        sorted_lengths, sorted_idx = data_length.sort(descending=True)
        conv_out = conv_out[sorted_idx]
        packed = pack_padded_sequence(conv_out, sorted_lengths.cpu(),
                                     batch_first=True, enforce_sorted=True)
        
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        attn_weights = self.attention(unpacked).squeeze(-1)
        mask = torch.arange(unpacked.size(1), device=attn_weights.device)[None, :] < sorted_lengths[:, None]
        attn_weights = attn_weights * mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-10)
        context = torch.bmm(attn_weights.unsqueeze(1), unpacked).squeeze(1)
        
        out = self.fc1(context)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.fc_drop1(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = F.relu(out)
        out = self.fc_drop2(out)
        
        return self.fc3(out)

@st.cache_resource
def load_prediction_model():
    """Load AFP prediction model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = igemTJModel().to(device)
    model.load_state_dict(torch.load('model/afp_model_920im_9493.pkl', map_location=device), strict=False)
    model.eval()
    return model, device

def coll_paddding(batch_traindata):
    """Data batching function"""
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature0 = []
    train_y = []

    for data in batch_traindata:
        feature0.append(data[0])
        train_y.append(data[1])
    data_length = [len(data) for data in feature0]
    feature0 = nn.utils.rnn.pad_sequence(feature0, batch_first=True, padding_value=0)
    return feature0, torch.tensor(train_y, dtype=torch.long), torch.tensor(data_length)

            

# ======================== Streamlit Application Interface ========================
def main():
    # Page configuration
    st.set_page_config(
        page_title="Antifreeze Protein Prediction System",
        page_icon="❄️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom styles
    st.markdown("""
    <style>
    .header {
        color: #1e3a8a;
        border-bottom: 2px solid #1e3a8a;
        padding-bottom: 10px;
    }
    .result-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .afp-card {
        background-color: #e0f2fe;
        border-left: 5px solid #0284c7;
    }
    .non-afp-card {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
    }
    .sequence-display {
        font-family: monospace;
        background-color: #f8fafc;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
    .progress-bar {
        height: 20px;
        background-color: #e5e7eb;
        border-radius: 10px;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background-color: #3b82f6;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    .warning {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application title
    st.title('❄️ Antifreeze Protein Prediction System')
    st.markdown("""
    Predict whether protein sequences are antifreeze proteins (AFPs) using deep learning models.  
    Upload FASTA files or directly input protein sequences for prediction.
    """)
    
    # Load models
    with st.spinner('Loading feature extraction model...'):
        tokenizer, feature_model, feature_device = load_feature_extractor()
    
    with st.spinner('Loading prediction model...'):
        pred_model, pred_device = load_prediction_model()
    
    st.success("Models loaded successfully!")
    
    # Threshold adjustment slider
    st.markdown("### Prediction Settings")
    threshold = st.slider("Adjust prediction threshold (0.3-0.7)", 
                         min_value=0.3, 
                         max_value=0.7, 
                         value=0.66, 
                         step=0.01,
                         help="Lower threshold increases AFP detection (more false positives). Higher threshold reduces false positives but may miss some AFPs.")
    
    # Input method selection
    input_method = st.radio("Select input method:", ("Input sequence", "Upload FASTA file"))
    
    sequences = []
    sequence_names = []
    
    if input_method == "Input sequence":
        seq_name = st.text_input("Sequence name (optional):", "Sequence_1")
        seq_input = st.text_area("Enter protein sequence (single-letter amino acid codes):", 
                               value="MNSFVVDILAFLHFLGLLLAGVAAQKVAGQAGDTSAPVGVGVGVGVG")
        
        if st.button("Predict"):
            if not seq_input.strip():
                st.warning("Please enter a protein sequence")
            else:
                # Validate sequence
                valid_seq = ''.join([c for c in seq_input.strip().upper() if c in "ACDEFGHIKLMNPQRSTVWYX"])
                if len(valid_seq) < 10:
                    st.error("Sequence too short or contains invalid characters. Use standard amino acid codes (20 standard + X).")
                else:
                    sequences = [valid_seq]
                    sequence_names = [seq_name]
    
    else:
        uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"])
        if uploaded_file is not None:
            # Read FASTA file
            fasta_data = uploaded_file.read().decode("utf-8")
            fasta_lines = fasta_data.split('\n')
            
            current_seq = ""
            current_name = ""
            for line in fasta_lines:
                if line.startswith(">"):
                    if current_seq and current_name:
                        sequences.append(current_seq)
                        sequence_names.append(current_name)
                        current_seq = ""
                    current_name = line[1:].strip()
                else:
                    current_seq += line.strip().upper()
            
            if current_seq and current_name:
                sequences.append(current_seq)
                sequence_names.append(current_name)
            
            if sequences:
                st.success(f"Successfully read {len(sequences)} sequences")
            else:
                st.warning("No valid sequences found")

        
        if sequences and results:
            with st.expander("Raw Prediction Data"):
                for idx, res in enumerate(results):
                    st.markdown(f"**Sequence {idx+1}**: {res['name']}")
                    st.code(f"""
                    Logits: {res['logits']}
                    Probability: {res['prob']}
                    """, language='python')
                
                
                csv = pd.DataFrame([{
                    'Sequence': res['name'],
                    'Logits': res['logits'],
                    'Probability': res['prob'],
                    'Prediction': 'AFP' if res['pred'] else 'Non-AFP'
                } for res in results])
                st.download_button(
                    label="Download Full Data",
                    data=csv.to_csv(index=False).encode('utf-8'),
                    file_name='prediction_data.csv',
                    mime='text/csv'
                )
    
    # Process and predict sequences
    if sequences:
        results = []
        st.subheader("Processing Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        total_seqs = len(sequences)
        
        # Add sequence length warning
        long_seqs = [seq for seq in sequences if len(seq) > 1000]
        if long_seqs:
            st.warning(f"Warning: {len(long_seqs)} long sequences (>1000 amino acids) detected. Processing may take longer.")
        
        # Process sequences individually
        for i, (name, seq) in enumerate(zip(sequence_names, sequences)):
            # Update status
            progress = (i + 1) / total_seqs
            status_text.text(f"Processing sequence {i+1}/{total_seqs}: {name} (Length: {len(seq)})")
            progress_bar.progress(progress)
            
            try:
                # Extract features
                emb = extract_prottran_features(seq, tokenizer, feature_model, feature_device)
                
                # Predict single sequence
                class TempDataset(Dataset):
                    def __init__(self, feature):
                        self.feature = [feature]
                    
                    def __len__(self):
                        return 1
                    
                    def __getitem__(self, idx):
                        return torch.tensor(self.feature[idx], dtype=torch.float32), 0
                
                dataset = TempDataset(emb)
                data_loader = DataLoader(
                    dataset, 
                    batch_size=1, 
                    shuffle=False,
                    collate_fn=coll_paddding
                )
                
                with torch.no_grad():
                    for data_x, _, data_length in data_loader:
                        outputs = pred_model(data_x.to(pred_device), data_length.to(pred_device))
                        logits_val = outputs.cpu().numpy()[0]
                        probs = torch.softmax(outputs, dim=1)
                        prob = probs[:, 1].item()
                
                        # Determine prediction result and confidence
                        prediction = "AFP" if prob > threshold else "Non-AFP"
                        
                        # Confidence calculation (based on distance from threshold)
                        conf_distance = abs(prob - threshold)
                        if conf_distance > 0.25:
                            confidence = "High confidence"
                        elif conf_distance > 0.15:
                            confidence = "Medium confidence"
                        else:
                            confidence = "Low confidence"

                        
                        results.append({
                            "Name": name,
                            "Sequence": seq,
                            "Length": len(seq),
                            "AFP Probability": prob,
                            "Prediction": prediction,
                            "Confidence": confidence,  
                            "Confidence Level": conf_distance,
                            "Logits": logits_val
                        })

                # Clean memory
                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    sleep(0.1)  # Short pause
                    
            except Exception as e:
                st.error(f"Error processing sequence '{name}': {str(e)}")
                results.append({
                    "Name": name,
                    "Sequence": seq,
                    "Length": len(seq),
                    "AFP Probability": 0.5,
                    "Prediction": "Error",
                    "Confidence": f"Processing failed",
                    "Confidence Level": 0,
                    "Logits": [0, 0] 
                })
        
        # Sort by confidence level
        results.sort(key=lambda x: x["Confidence Level"], reverse=True)
        
        # Display results
        st.subheader("Prediction Results")
        
        for result in results:
            # Error handling display
            if result["Prediction"] == "Error":
                with st.container():
                    st.markdown(f"<div class='result-card warning'>", unsafe_allow_html=True)
                    st.error(f"Sequence '{result['Name']}' processing failed")
                    st.markdown("</div>", unsafe_allow_html=True)
                continue
            
            # Normal result display
            is_afp = result["Prediction"] == "AFP"
            card_class = "afp-card" if is_afp else "non-afp-card"
            
            with st.container():
                st.markdown(f"<div class='result-card {card_class}'>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.subheader(result["Name"])
                    st.write(f"**Length:** {result['Length']} amino acids")
                    st.write(f"**Prediction:** {'AFP' if is_afp else 'Non-AFP'}")
                    st.write(f"**Confidence:** {result['Confidence']}")
                    
                    # Display probability progress bar
                    prob = result['AFP Probability']
                    st.write(f"**AFP Probability:** {prob:.4f}")
                    st.markdown(f"""
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {prob*100}%">
                            {prob*100:.1f}%
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 5px; font-size: 0.8em;">
                        Threshold: {threshold*100:.0f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.write("**Sequence Preview:**")
                    # Display sequence fragment (smart truncation)
                    preview_length = min(100, len(result['Sequence']))
                    preview = result['Sequence'][:preview_length]
                    if len(result['Sequence']) > preview_length:
                        preview += "..."
                    
                    with st.expander("View model raw outputs (logits)"):
                        logits = result.get('Logits', [0, 0])
                        st.write(f"**Non-AFP logit:** {logits[0]:.4f}")
                        st.write(f"**AFP logit:** {logits[1]:.4f}")
                    
                    # Display full sequence
                    with st.expander("View full sequence"):
                        st.text(result['Sequence'])
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Result statistics
        afp_count = sum(1 for r in results if r.get("Prediction") == "AFP")
        non_afp_count = sum(1 for r in results if r.get("Prediction") == "Non-AFP")
        error_count = sum(1 for r in results if r.get("Prediction") == "Error")
        
        st.success(f"**Prediction Summary**: AFP: {afp_count} | Non-AFP: {non_afp_count} | Errors: {error_count}")
        
        # Download results
        if results:
            # Create download data
            download_data = []
            for res in results:
                download_data.append({
                    "Sequence Name": res["Name"],
                    "Length": res["Length"],
                    "AFP Probability": res["AFP Probability"],
                    "Prediction": res["Prediction"],
                    "Confidence": res["Confidence"].split(" ")[0],  # Remove icon
                    "Sequence": res["Sequence"]
                })
            
            df = pd.DataFrame(download_data)
            csv = df.to_csv(index=False)
            
            # Add timestamp to filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"afp_predictions_{timestamp}.csv"
            
            st.download_button(
                label="Download Prediction Results (CSV)",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
    
    # About section
    st.sidebar.title("About")
    st.sidebar.info("""
    **Antifreeze Protein Prediction System**  
    Predicts whether proteins are antifreeze proteins (AFPs) using deep learning models.
    
    **Technical Details:**  
    - Feature Extraction: ProtT5-XL-UniRef50 model
    - Prediction Model: Bidirectional LSTM + Fully Connected Network  
    - Training Data: 920 imbalanced AFP and non-AFP sequences  
    
    **How to Use:**  
    1. Input protein sequence or upload FASTA file  
    2. Click "Predict" button  
    3. View prediction results and probabilities  
    4. Download full prediction results
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    **Note:**  
    - Sequence length limited to 2500 amino acids  
    - Long sequences may require significant processing time  
    - GPU acceleration recommended for optimal performance
    """)

if __name__ == "__main__":
    main()