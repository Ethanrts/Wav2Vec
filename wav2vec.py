from datasets import load_dataset, Dataset, Audio, load_from_disk
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config, Wav2Vec2FeatureExtractor, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import evaluate

labels_path = '/content/drive/MyDrive/244 Final Project/train_set_gold.csv'
output_path = '/content/drive/MyDrive/244 Final Project/output_4'

gold_df = pd.read_csv(labels_path)

idx_to_label = {i:label for i, label in enumerate(gold_df['label'].unique())}
label_to_idx = {v:k for k, v in idx_to_label.items()}

encoded_gold = [label_to_idx[label] for label in gold_df['label']]

dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", streaming=True, trust_remote_code=True)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

train_inputs = []
sampling_rates = []
num_load = 10000
for i, data in enumerate(dataset):
  if i < num_load:
    train_inputs.append(data['audio']['array'])
    sampling_rates.append(data['audio']['sampling_rate'])
  else:
    break

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", sampling_rate=sampling_rates[0])

inputs = feature_extractor(train_inputs, sampling_rate=sampling_rates[0])

train_data = {
    'input_values': inputs.input_values,
    'labels': encoded_gold
}

train_data['labels'][0]

loaded_val_data = load_from_disk("/content/drive/MyDrive/244 Final Project/val_data")

batch_size = 4
lr = 1e-5
num_epochs = 5

train_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    num_train_epochs=num_epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch"
    )

model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=len(idx_to_label))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_score(y_true=labels, y_pred=predictions)


pretrainer = Trainer(
    model,
    tokenizer=feature_extractor,
    args=train_args,
    train_dataset=loaded_val_data,
    eval_dataset=loaded_val_data,
    compute_metrics=compute_metrics
    )

pretrainer.train()