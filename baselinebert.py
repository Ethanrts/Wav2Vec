import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
samples_data = pd.read_csv('samples.csv')
gold_data = pd.read_csv('train_set_gold.csv')

# Merge the datasets based on the 'ID' column
combined_data = pd.merge(samples_data, gold_data[['ID', 'label']], on='ID')

# Tokenize and prepare data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(combined_data['sentences'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
labels = combined_data['label']

print(labels)

# Convert labels to numeric representation
label_mapping = {label: i for i, label in enumerate(set(labels))}
labels = [label_mapping[label] for label in labels]

train_indices, val_indices, _, _ = train_test_split(range(len(tokens['input_ids'])), labels, test_size=0.2, random_state=42)

# Create DataLoader for training and validation
train_dataset = TensorDataset(tokens['input_ids'][train_indices], tokens['attention_mask'][train_indices], torch.tensor(labels)[train_indices])
val_dataset = TensorDataset(tokens['input_ids'][val_indices], tokens['attention_mask'][val_indices], torch.tensor(labels)[val_indices])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

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
model.save_pretrained('trained_bert_model')

# Evaluation on validation set
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, label = batch
        input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(label.cpu().numpy())

# Calculate accuracy and F1 score
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')  # Adjust 'average' as needed

print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Weighted F1 Score: {f1 * 100:.2f}%')

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)

# Display confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
