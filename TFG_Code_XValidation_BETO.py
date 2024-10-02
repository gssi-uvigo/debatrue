import torch
import pandas as pd
import random
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef

# Set the seed for reproducibility
seed = 26
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the dataset from the CSV file
df = pd.read_csv("/content/D600.csv", delimiter=";")

# Split the data into features (X) and labels (y)
X = df[["Titulo", "Descripcion", "Fecha"]]
y = df["Label"]

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Initialize variables to keep track of the best model
best_mcc = -1.0
best_epoch = -1
best_model_path = "/content/drive/MyDrive/DATASET/Borrar"

# Number of epochs
n_epochs = 10

# Create StratifiedKFold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the epochs
for epoch in range(n_epochs):  # 10 epochs
    print(f"Epoch {epoch + 1}/{n_epochs}")
    
    # Each epoch will have a new 5-fold split
    for fold, (train_indices, eval_indices) in enumerate(skf.split(X, y)):
        X_train, X_eval = X.iloc[train_indices], X.iloc[eval_indices]
        y_train, y_eval = y.iloc[train_indices], y.iloc[eval_indices]

        # Encode the features of the training and evaluation sets
        train_encodings = tokenizer(
            X_train["Titulo"].tolist(),
            X_train["Descripcion"].tolist(),
            X_train["Fecha"].tolist(),
            padding="max_length",
            truncation='only_second',
            max_length=128,
            return_tensors="pt"
        )
        eval_encodings = tokenizer(
            X_eval["Titulo"].tolist(),
            X_eval["Descripcion"].tolist(),
            X_eval["Fecha"].tolist(),
            padding="max_length",
            truncation='only_second',
            max_length=128,
            return_tensors="pt"
        )

        # Assign the encoded inputs to separate variables
        train_input_ids = train_encodings["input_ids"]
        train_attention_masks = train_encodings["attention_mask"]
        eval_input_ids = eval_encodings["input_ids"]
        eval_attention_masks = eval_encodings["attention_mask"]

        # Create TensorDatasets
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, torch.tensor(y_train.tolist()))
        eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, torch.tensor(y_eval.tolist()))

        # Create DataLoaders to load data in batches
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # Load the pre-trained model
        model = RobertaForSequenceClassification.from_pretrained("PlanTL-GOB-ES/roberta-base-bne", num_labels=2)

        # Configure the optimizer and training device
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set the dropout rate
        dropout_rate = 0.5
        model.classifier.dropout.p = dropout_rate

        # Model training
        model.train()

        total_train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Print the first line of the training set
        print("First training data point:", train_dataloader)

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss

            logits = outputs.logits

            total_train_loss += loss.item()

            _, predicted_labels = torch.max(logits, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

            loss.backward()
            optimizer.step()

        train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_predictions / total_predictions

        # Evaluation on the evaluation set
        model.eval()

        with torch.no_grad():
            total_eval_loss = 0.0
            eval_predictions = []
            eval_labels = []

            for batch in eval_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_masks, labels = batch

                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_eval_loss += loss.item()

                _, predicted_labels = torch.max(logits, 1)
                eval_predictions.extend(predicted_labels.tolist())
                eval_labels.extend(labels.tolist())

            eval_loss = total_eval_loss / len(eval_dataloader)
            eval_accuracy = accuracy_score(eval_labels, eval_predictions)
            eval_f1 = f1_score(eval_labels, eval_predictions)
            eval_recall = recall_score(eval_labels, eval_predictions)
            eval_mcc = matthews_corrcoef(eval_labels, eval_predictions)

        print(f"Fold {fold + 1}")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Eval Loss: {eval_loss:.4f} | Eval Accuracy: {eval_accuracy:.4f}")
        print(f"Eval F1: {eval_f1:.4f}")
        print(f"Eval Recall: {eval_recall:.4f}")
        print(f"Eval MCC: {eval_mcc:.4f}")
        print("--------------------")

        # Save the model if a higher MCC is achieved
        if eval_mcc > best_mcc:
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            best_mcc = eval_mcc
            best_epoch = epoch + 1

# Print results of the best model
print("Best model achieved at epoch:", best_epoch)
print("Best evaluation MCC:", best_mcc)
print("Model saved at:", best_model_path)
