import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved model
model = BertForSequenceClassification.from_pretrained('trained_bert_model', num_labels=5)

# Load the validation data
val_data = pd.read_csv('val_sentences.csv')

# Tokenize and prepare validation data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
val_tokens = tokenizer(val_data['sentences'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)

# Prediction on the entire validation set
model.eval()
predictions = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    for batch in DataLoader(TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask']), batch_size=8, shuffle=False):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

print(predictions[:10])

# Assuming you have true labels for the validation set
true_labels_data = pd.read_csv('val_labelled.csv')
true_labels = true_labels_data['label']

print(true_labels[:10])
# Assuming label_mapping was used during training
label_mapping = {'idea': 0, 'thing': 1, 'person': 2, 'not applicable': 3, 'place': 4}


print(label_mapping)

# Convert labels to numeric representation using label_mapping
true_labels_numeric = [label_mapping[label] for label in true_labels]

# Generate confusion matrix for the validation set
conf_matrix = confusion_matrix(true_labels_numeric, predictions)


# Display confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Validation Set')
plt.show()
