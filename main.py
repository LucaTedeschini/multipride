import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
    AutoConfig,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset as HFDataset
from evaluate import load as load_metric
from huggingface_hub.utils import disable_progress_bars
import os
import gc
import numpy as np
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from rich.console import Console
from rich.table import Table
from rich import print
import questionary
import sys

### Setup ###
console = Console()
disable_progress_bars()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Twitter/twhin-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

###########################################################
### Stage 1: Pre-train on LGBT attribute identification ###
###########################################################

console.print(
    "[bold underline purple]Starting Stage 1: Pre-training for LGBT attribute[/bold underline purple]"
)

## Data Loading ##
console.print("[bold]Loading augmented datasets...[/bold]")
ita = pd.read_csv("dataset/augmented_it.csv")
esp = pd.read_csv("dataset/augmented_es.csv")
console.print("[bold green]All datasets available.[/bold green]")

dataset_map = {
    "Italian": ita,
    "Spanish": esp,
}

selected_languages = questionary.checkbox(
    "Select on which dataset(s) you want to train the model:",
    choices=list(dataset_map.keys())
).ask()

if not selected_languages:
    console.print("[bold red]No datasets selected. Exiting...[/bold red]")
    sys.exit()
else:
    console.print(f"\n:white_check_mark: [bold green]Selected:[/bold green] {', '.join(selected_languages)}\n")

    datasets_to_train = [dataset_map[lang] for lang in selected_languages]
    
    dataset = pd.concat(datasets_to_train, ignore_index=True)
    dataset['bio'] = dataset['bio'].fillna('')
    console.print(f"[bold green]Datasets loaded and combined. Total length: [cyan]{len(dataset)}[/cyan][/bold green]")

## Data Splitting ##
pre_train_df, pre_test_df = train_test_split(
    dataset, test_size=0.3, stratify=dataset["lgbt"], random_state=42
)

## Tokenization ##
def tokenize(batch):
    return tokenizer(
        batch["text"],
        batch["bio"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_ds = HFDataset.from_pandas(pre_train_df).map(tokenize, batched=True)
test_ds = HFDataset.from_pandas(pre_test_df).map(tokenize, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "lgbt"], output_all_columns=True)
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "lgbt"], output_all_columns=True)


## Class Weight Computation ##
label_counts = pre_train_df["lgbt"].value_counts().to_dict()
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in sorted(label_counts.keys())]
class_weights = torch.tensor(weights, dtype=torch.float)
console.print(f"[bold][green]Class weights for pre-training: [/green]{class_weights}[/bold]")

## Model and Trainer Setup ##
console.print("[bold yellow]Loading model for pre-training...[/bold yellow]")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
console.print("[bold green]Model loaded![/bold green]")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels)["f1"],
    }

## Training ##
training_args = TrainingArguments(
    output_dir="./results_lgbt_pretrain",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs_lgbt_pretrain",
    logging_steps=50,
    save_total_limit=2,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

console.print("\n[bold yellow]Starting pre-training...[/bold yellow]")
trainer.train()
console.print("[bold green]:white_check_mark: Pre-training complete.[/bold green]")


########################################################
### Stage 2: Dual Encoder for Reclamatory classifier ###
########################################################

console.print(
    "\n[bold underline purple]Starting Stage 2: Dual Encoder for Reclamatory Classification[/bold underline purple]"
)

## Dual Encoder Model Definition ##
class DualEncoderForSequenceClassification(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder_text = AutoModel.from_config(config)
        self.encoder_bio = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs_text = self.encoder_text(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        outputs_bio = self.encoder_bio(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)

        h_text = outputs_text.last_hidden_state[:, 0]
        h_bio = outputs_bio.last_hidden_state[:, 0]

        combined = torch.cat((h_text, h_bio), dim=-1)
        gate = self.gate_layer(combined)
        h_final = gate * h_text + (1 - gate) * h_bio

        pooled_output = self.dropout(h_final)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            class_weights = torch.tensor(self.config.class_weights, device=self.device) if hasattr(self.config, 'class_weights') and self.config.class_weights is not None else None
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits)

## Main Task Data Preparation ##
train_df, test_df = train_test_split(
    dataset, test_size=0.3, stratify=dataset["label"], random_state=42
)

train_ds = HFDataset.from_pandas(train_df).map(tokenize, batched=True)
test_ds = HFDataset.from_pandas(test_df).map(tokenize, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

label_counts = train_df["label"].value_counts().to_dict()
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in sorted(label_counts.keys())]
class_weights = torch.tensor(weights, dtype=torch.float)
console.print(f"[bold][green]Class weights for main task: [/green]{class_weights}[/bold]")

## Model Initialization & Weight Loading ##
console.print("[bold]Initializing Dual Encoder model...[/bold]")
config = AutoConfig.from_pretrained(model_name, num_labels=2)
config.class_weights = class_weights.tolist()
combined_model = DualEncoderForSequenceClassification(config)

base_model = AutoModel.from_pretrained(model_name)
combined_model.encoder_text.load_state_dict(base_model.state_dict())
console.print("[cyan]-> Text encoder loaded with base weights.[/cyan]")

state_dict = trainer.model.state_dict()
filtered_state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items() if k.startswith("bert.")}
combined_model.encoder_bio.load_state_dict(filtered_state_dict, strict=False)
console.print("[cyan]-> Bio encoder loaded with pre-trained LGBT classifier weights.[/cyan]")

for param in combined_model.encoder_bio.parameters():
    param.requires_grad = False
console.print("[yellow]Bio encoder layers frozen.[/yellow]")

del trainer.model, trainer, base_model
gc.collect()
torch.cuda.empty_cache()
combined_model.to(device)

## Main Task Training ##
training_args = TrainingArguments(
    output_dir="./results_dual_encoder",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs_dual_encoder",
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=combined_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

console.print("\n[bold yellow]Starting main task training...[/bold yellow]")
trainer.train()
console.print("[bold green]:white_check_mark: Main task training complete.[/bold green]")

#########################
### Results Analysis ###
#########################
console.print("\n[bold underline purple]Starting Results Analysis[/bold underline purple]")

## Prediction and Metrics ##
predictions_output = trainer.predict(test_ds)
table = Table(title="Test Set Metrics")
table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
table.add_column("Value", justify="right", style="magenta")

for k, v in predictions_output.metrics.items():
    table.add_row(k.replace("test_", "").capitalize(), f"{v:.4f}" if isinstance(v, float) else str(v))
print(table)

## Error Analysis ##
logits = predictions_output.predictions
true_labels = predictions_output.label_ids
predicted_labels = np.argmax(logits, axis=-1)
probabilities = softmax(logits, axis=1)
confidence_scores = np.max(probabilities, axis=1)

results_df = test_df.copy()
results_df['predicted_label'] = predicted_labels
results_df['true_label'] = true_labels
results_df['confidence'] = confidence_scores
results_df['is_correct'] = (results_df['true_label'] == results_df['predicted_label'])

output_filename = "error_analysis_results.csv"
results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
console.print(f"\n[bold green]Error analysis results saved to [cyan]'{output_filename}'[/cyan][/bold green]")

## Confusion Matrix ##
console.print("\n[bold]Confusion Matrix:[/bold]")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)
class_labels = ['Non-Reclamatory', 'Reclamatory']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig("confusionmatrix.png")