import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data for training
train_samples_data = pd.read_csv('samples.csv')
train_gold_data = pd.read_csv('train_set_gold.csv')

# Merge the training datasets based on the 'ID' column
train_combined_data = pd.merge(train_samples_data, train_gold_data[['ID', 'label']], on='ID')

# Tokenize and prepare training data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_tokens = tokenizer(train_combined_data['sentences'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
train_labels = train_combined_data['label']

# Convert labels to numeric representation
label_mapping = {label: i for i, label in enumerate(set(train_labels))}
label_mapping_inv = {i: label for label, i in label_mapping.items()}  # Added line to create inverse mapping
train_labels = [label_mapping[label] for label in train_labels]

train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], torch.tensor(train_labels))
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_labels)))

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        input_ids, attention_mask, label = batch
        input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Save the trained model
model.save_pretrained('trained_model')

# Load the trained model
model = BertForSequenceClassification.from_pretrained('trained_model', num_labels=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the validation data and true labels
val_data = pd.read_csv('val_sentences.csv')
val_label_data = pd.read_csv('val_labelled.csv')

# Merge the validation datasets based on the 'ID' column
val_combined_data = pd.merge(val_data, val_label_data[['ID', 'label']], on='ID')

# Tokenize and prepare validation data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
val_tokens = tokenizer(val_combined_data['sentences'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
val_labels = val_combined_data['label']

# Convert labels to numeric representation
val_labels = [label_mapping[label] for label in val_labels]

# Prediction on the entire validation set
model.eval()
predictions = []

with torch.no_grad():
    for batch in DataLoader(TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask']), batch_size=8, shuffle=False):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

# Save or use predicted_labels as needed
print(predictions)

# Calculate accuracy and F1 score
accuracy = accuracy_score(val_labels, predictions)
f1 = f1_score(val_labels, predictions, average='weighted')  # Adjust 'average' as needed

print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Weighted F1 Score: {f1 * 100:.2f}%')

# Generate confusion matrix for validation set
conf_matrix = confusion_matrix(val_labels, predictions)

# Display confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Validation Set')
plt.show()