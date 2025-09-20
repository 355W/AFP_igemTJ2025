# AFP_IGEMTianjin2025
The package includes an executable to identify Antifreeze Proteins for given sequences. 

It also includes all components of AFP_IGEMTianjin2025. You may incorporate AFP_IGEMTianjin2025 into your pipeline or modify it to suit your needs.

Please also check out our webserver for real-time prediction of antifreeze protein features. https://gitlab.igem.org/2025/software-tools/tianjin
# Description
This antifreeze protein (AFP) prediction model presents an end-to-end computational framework for predicting protein function directly from sequence data. It employs a two-stage architecture combining a pre-trained protein language model for feature extraction with a bespoke deep learning classifier for accurate functional annotation.

The model is built upon the ProtT5-XL-UniRef50 encoder, a transformer-based model pre-trained on the UniRef50 database, to generate high-dimensional feature representations (1024 dimensions per amino acid) from input protein sequences. These rich embeddings capture complex semantic and syntactic patterns within the protein sequences. The subsequent classifier, named igemTJModel, utilizes a sophisticated multi-modal neural network architecture to process these features. It integrates 1D convolutional layers (1D-CNN) to identify local amino acid motifs, bidirectional LSTM (Bi-LSTM) layers to capture long-range contextual dependencies across the sequence, and an attention mechanism to dynamically weigh the importance of specific residues, significantly enhancing model interpretability. Regularization strategies like Layer Normalization, Dropout, and residual connections are incorporated to ensure training stability and prevent overfitting.

Additionally, a user-friendly web application is built on Streamlit for real-time prediction and hypothesis testing by researchers. This tool provides a powerful, high-performance, and interpretable platform for accelerating AFP discovery and analysis.
# Dataset
AFP920.seq: this file contains 920 AFPs with key-value format

Non-AFP9493.seq: this file contains 9493 Non-AFPs with key-value format

We constructed a dataset comprising 920 AFPs and 9493 non-AFPs.
A comprehensive search was conducted in the UniProtKB database until January 24, 2025 using specific keywords, resulting in the collection of 6589 AFPs. Next, the maximal pairwise sequence identity of the proteins in the manually inspected dataset was culled to ≤40 % using CD-HIT, yielding a set of 920 unique AFPs.

The negative dataset was derived from 9493 seed proteins of Pfam protein families that are not associated with antifreeze proteins , which is widely used for evaluating the performance of AFPs prediction methods.

Balance dataset: The dataset was divided into training and test sets, 644 AFPs and 644 non-AFPs were randomly selected as positive and negative samples to form the training dataset, and the remaining 276 AFPs and 8849 non-AFPs were designated as the test dataset

Imbalanced dataset: To further validate the predictive performance and facilitate subsequent research, we introduced an imbalanced dataset. This dataset was divided with 70% AFPs and non-AFPs for training and the remaining 30% for independent testing respectively.
# Installation
## Set up environment for AFP_IGEMTianjin2025
Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master.
```
pip install requirements
```
## Extract features
Extract pLMs embedding: cd to the AFP_IGEMTianjin2025 dictionary, and run "python3 ProtT5_Extraction.py", the pLMs embedding matrixs will be extracted to midData/ProtTran fold.
## Model training and testing
Cd to the AFP_IGEMTianjin2025 dictionary,and run "python3 AFP-pLMs_920im.py"
The model will be trained and tested, and the results will be saved in the result fold.
# Usage
After running `streamlit run afp_pre.py`, the following steps will be displayed in the web interface:
## Model Loading: 
The application will first load the feature extraction (ProtT5) and prediction models. A progress indicator will show the loading status.
## Prediction Settings: 
Adjust the prediction threshold using a slider (default is 0.66). Lower values increase sensitivity for AFP detection (more false positives), while higher values reduce false positives but may miss some AFPs.
## Input Method:
Option 1: Input Sequence: Enter a protein sequence manually (single-letter amino acid codes) and an optional sequence name.

Option 2: Upload FASTA File: Upload a FASTA file containing one or multiple protein sequences.
## Prediction Execution: 
Click the "Predict" button to start processing. A progress bar will show the status of sequence feature extraction and prediction.
## Results Display:
Prediction Execution: Click the "Predict" button to start processing. A progress bar will show the status of sequence feature extraction and prediction.

Predictions are shown as individual cards with:
`Sequence name/length`
`AFP/Non-AFP prediction`
`Confidence level (High/Medium/Low)`
`AFP probability visual progress bar`
Expandable sections for raw model outputs (`logits`) and full sequences
`Results` are sorted by confidence level
## Download Options:
Download all prediction results as a `CSV` file with details including sequence names, lengths, probabilities, and predictions.

# Contributing
We welcome contributions! Please contact us first to discuss changes you’d like to make. To get started:

Fork the repository.

Make your changes.

Create a pull request.

Invite us to review your changes.

# Authors and acknowledgment
The project was developed by:

Yanwen Li

Fengyuan Liu

Renkui Wen








