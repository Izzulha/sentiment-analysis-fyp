import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import random
import time
import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ENHANCED CONFIGURATION & HYPERPARAMETERS
# ==========================================
config = {
    'seed_val': 42,
    'batch_size': 16,                      # Optimal for BERT fine-tuning
    'epochs': 5,                           # Increased for better learning
    'learning_rate': 2e-5,                 # Optimal LR for BERT
    'max_len': 256,                        # Increased for more context
    'dropout_rate': 0.2,                   # Increased to prevent overfitting
    'weight_decay': 0.01,
    'model_name': 'bert-base-multilingual-cased',
    'data_path': 'mbert modeling data.csv',
    'gradient_accumulation_steps': 1,
    'warmup_steps': 100,
    'epsilon': 1e-8,
    'use_class_weights': True,             # Handle class imbalance
    'use_early_stopping': True,            # Prevent overfitting
    'early_stopping_patience': 3,          # Patience for early stopping
    'use_focal_loss': False,               # For severe imbalance
    'save_best_model': True,               # Save best model during training
    'adam_epsilon': 1e-8,
    'grad_clip': 1.0,                      # Gradient clipping value
}

# Set all seeds for reproducibility
def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config['seed_val'])

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print("\n" + "="*60)
print("LOADING AND PREPROCESSING DATA")
print("="*60)

# Load data
if not os.path.exists(config['data_path']):
    print(f"ERROR: File '{config['data_path']}' not found!")
    print("Please ensure the CSV file is in the correct directory.")
    exit()

try:
    df = pd.read_csv(config['data_path'], encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(config['data_path'], encoding='latin-1')
    except:
        df = pd.read_csv(config['data_path'], encoding='ISO-8859-1')

print(f"Loaded {len(df)} records from dataset")

# Data cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep basic punctuation
    import re
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

# Clean the text
df['ENGLISH'] = df['ENGLISH'].apply(clean_text)

# Remove empty texts
df = df[df['ENGLISH'].str.len() > 0]
print(f"After cleaning: {len(df)} records")

# Map sentiment labels
label_map = {'Negative': 0, 'Positive': 1, 'negative': 0, 'positive': 1}
df = df[df['Sentiment'].isin(label_map.keys())]
df['label'] = df['Sentiment'].map(label_map)

# Check class distribution
print("\nCLASS DISTRIBUTION:")
class_counts = df['label'].value_counts()
for label, count in class_counts.items():
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"  {sentiment}: {count} samples ({count/len(df)*100:.1f}%)")

# Calculate class weights for imbalance handling
class_weights = compute_class_weight(
    'balanced',
    classes=np.array([0, 1]),
    y=df['label'].values
)
print(f"\nClass weights for imbalance: {class_weights}")

texts = df['ENGLISH'].values
labels = df['label'].values

# ==========================================
# 3. TOKENIZATION
# ==========================================
print(f"\nLoading {config['model_name']} tokenizer...")
tokenizer = BertTokenizer.from_pretrained(config['model_name'])

def preprocess_data(texts, labels, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    
    print(f"Tokenizing {len(texts)} samples...")
    
    for i, text in enumerate(texts):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(texts)} samples...")
        
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, labels

# Preprocess all data
input_ids, attention_masks, labels = preprocess_data(
    texts, labels, tokenizer, config['max_len']
)

# ==========================================
# 4. DATA SPLITTING WITH STRATIFICATION
# ==========================================
print("\n" + "="*60)
print("CREATING TRAIN/VALIDATION/TEST SPLITS")
print("="*60)

# First split: 80% train, 20% temp
train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(
    input_ids, labels,
    test_size=0.2,
    random_state=config['seed_val'],
    stratify=labels.numpy()
)
train_masks, temp_masks, _, _ = train_test_split(
    attention_masks, labels,
    test_size=0.2,
    random_state=config['seed_val'],
    stratify=labels.numpy()
)

# Second split: 10% validation, 10% test
val_inputs, test_inputs, val_labels, test_labels = train_test_split(
    temp_inputs, temp_labels,
    test_size=0.5,
    random_state=config['seed_val'],
    stratify=temp_labels.numpy()
)
val_masks, test_masks, _, _ = train_test_split(
    temp_masks, temp_labels,
    test_size=0.5,
    random_state=config['seed_val'],
    stratify=temp_labels.numpy()
)

print(f"\nDATASET SPLIT SUMMARY:")
print(f"  Training samples: {len(train_inputs)}")
print(f"  Validation samples: {len(val_inputs)}")
print(f"  Test samples: {len(test_inputs)}")
print(f"  Total samples: {len(train_inputs) + len(val_inputs) + len(test_inputs)}")

print("\nCLASS DISTRIBUTION IN EACH SPLIT:")
for name, labels_split in [('Train', train_labels), ('Validation', val_labels), ('Test', test_labels)]:
    unique, counts = np.unique(labels_split.numpy(), return_counts=True)
    dist_dict = {int(u): int(c) for u, c in zip(unique, counts)}
    print(f"  {name}: {dist_dict}")

# ==========================================
# 5. CREATE DATALOADERS
# ==========================================
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=config['batch_size'],
    num_workers=2 if torch.cuda.is_available() else 0,
    pin_memory=True if torch.cuda.is_available() else False
)

validation_data = TensorDataset(val_inputs, val_masks, val_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(
    validation_data,
    sampler=validation_sampler,
    batch_size=config['batch_size']
)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(
    test_data,
    sampler=test_sampler,
    batch_size=config['batch_size']
)

# ==========================================
# 6. LOAD MODEL WITH ENHANCED SETTINGS
# ==========================================
print(f"\nLoading {config['model_name']} model...")
model = BertForSequenceClassification.from_pretrained(
    config['model_name'],
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    hidden_dropout_prob=config['dropout_rate'],
    attention_probs_dropout_prob=config['dropout_rate'],
)

model.to(device)

# ==========================================
# 7. SETUP OPTIMIZER & SCHEDULER
# ==========================================
print("\nSetting up optimizer and scheduler...")

# Differential learning rates for different parameter groups
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() 
                  if not any(nd in n for nd in no_decay)],
        'weight_decay': config['weight_decay'],
    },
    {
        'params': [p for n, p in model.named_parameters() 
                  if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
    }
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=config['learning_rate'],
    eps=config['adam_epsilon']
)

# Learning rate scheduler with warmup
total_steps = len(train_dataloader) * config['epochs']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),  # 10% warmup
    num_training_steps=total_steps
)

# Class weights for loss function
if config['use_class_weights']:
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
else:
    loss_fn = None  # Use default from model

# ==========================================
# 8. HELPER FUNCTIONS
# ==========================================
def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def calculate_metrics(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    accuracy = accuracy_score(labels_flat, pred_flat)
    precision = precision_score(labels_flat, pred_flat, average='weighted')
    recall = recall_score(labels_flat, pred_flat, average='weighted')
    f1 = f1_score(labels_flat, pred_flat, average='weighted')
    
    return accuracy, precision, recall, f1

# ==========================================
# 9. TRAINING LOOP WITH EARLY STOPPING
# ==========================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

training_stats = []
total_t0 = time.time()

best_val_accuracy = 0
patience_counter = 0
best_model_state = None

for epoch_i in range(config['epochs']):
    print(f'\n{"="*60}')
    print(f'EPOCH {epoch_i + 1}/{config["epochs"]}')
    print(f'{"="*60}')
    
    # ========== TRAINING ==========
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step}/{len(train_dataloader)}. Elapsed: {elapsed}')
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        
        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels
        )
        
        loss = outputs.loss
        if loss_fn is not None:
            logits = outputs.logits
            loss = loss_fn(logits, b_labels)
        
        total_train_loss += loss.item()
        
        # Calculate training accuracy
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    
    print(f'\n  Average training loss: {avg_train_loss:.4f}')
    print(f'  Training accuracy: {avg_train_accuracy:.4f}')
    print(f'  Training time: {training_time}')
    
    # ========== VALIDATION ==========
    print('\nRunning Validation...')
    t0 = time.time()
    model.eval()
    
    total_val_loss = 0
    total_val_accuracy = 0
    all_val_preds = []
    all_val_labels = []
    
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            outputs = model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels
            )
        
        loss = outputs.loss
        total_val_loss += loss.item()
        
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        
        total_val_accuracy += flat_accuracy(logits, label_ids)
        all_val_preds.append(logits)
        all_val_labels.append(label_ids)
    
    avg_val_loss = total_val_loss / len(validation_dataloader)
    avg_val_accuracy = total_val_accuracy / len(validation_dataloader)
    
    # Calculate additional metrics
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)
    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(val_preds, val_labels)
    
    print(f'\n  Validation Loss: {avg_val_loss:.4f}')
    print(f'  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)')
    print(f'  Validation Precision: {val_precision:.4f}')
    print(f'  Validation Recall: {val_recall:.4f}')
    print(f'  Validation F1 Score: {val_f1:.4f}')
    
    # ========== EARLY STOPPING ==========
    if config['use_early_stopping']:
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f'  ✅ NEW BEST MODEL! Validation Accuracy: {val_accuracy:.4f}')
        else:
            patience_counter += 1
            print(f'  Early stopping patience: {patience_counter}/{config["early_stopping_patience"]}')
            
            if patience_counter >= config['early_stopping_patience']:
                print(f'\n⚠️  Early stopping triggered after {epoch_i + 1} epochs')
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    # Store statistics
    training_stats.append({
        'epoch': epoch_i + 1,
        'Training Loss': avg_train_loss,
        'Training Accuracy': avg_train_accuracy,
        'Validation Loss': avg_val_loss,
        'Validation Accuracy': val_accuracy,
        'Validation Precision': val_precision,
        'Validation Recall': val_recall,
        'Validation F1': val_f1,
        'Training Time': training_time,
    })

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print(f"Total training time: {format_time(time.time() - total_t0)}")
print("="*60)

# ==========================================
# 10. FINAL EVALUATION ON TEST SET
# ==========================================
print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

model.eval()

all_test_preds = []
all_test_labels = []
all_test_logits = []

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
    
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.cpu().numpy()
    
    all_test_logits.append(logits)
    all_test_labels.append(label_ids)
    all_test_preds.append(np.argmax(logits, axis=1).flatten())

# Flatten all predictions and labels
flat_predictions = np.concatenate(all_test_preds, axis=0)
flat_true_labels = np.concatenate(all_test_labels, axis=0)
flat_logits = np.concatenate(all_test_logits, axis=0)

# Calculate metrics
accuracy = accuracy_score(flat_true_labels, flat_predictions)
precision = precision_score(flat_true_labels, flat_predictions, average='weighted')
recall = recall_score(flat_true_labels, flat_predictions, average='weighted')
f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')

# Detailed classification report
target_names = ['Negative', 'Positive']
report = classification_report(
    flat_true_labels, 
    flat_predictions, 
    target_names=target_names,
    digits=4
)

# Confusion Matrix
cm = confusion_matrix(flat_true_labels, flat_predictions)

print(f"\n📊 FINAL TEST RESULTS:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

print("\n📈 CLASSIFICATION REPORT:")
print(report)

print("\n🎯 CONFUSION MATRIX:")
print(f"[[TN: {cm[0,0]}, FP: {cm[0,1]}]")
print(f" [FN: {cm[1,0]}, TP: {cm[1,1]}]]")

# ==========================================
# 11. SAVE MODEL AND RESULTS
# ==========================================
print("\n" + "="*60)
print("SAVING MODEL AND RESULTS")
print("="*60)

# Create output directory
output_dir = './bert_sentiment_model_high_accuracy/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model saved to: {output_dir}")

# Save training statistics
stats_df = pd.DataFrame(training_stats)
stats_df.to_csv(os.path.join(output_dir, 'training_stats.csv'), index=False)
print("✅ Training statistics saved to: training_stats.csv")

# Save test results
test_results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'best_validation_accuracy': float(best_val_accuracy),
    'predictions': flat_predictions.tolist(),
    'true_labels': flat_true_labels.tolist(),
    'confusion_matrix': cm.tolist(),
    'config': config
}

with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
    json.dump(test_results, f, indent=2)
print("✅ Test results saved to: test_results.json")

# Save prediction samples for analysis
prediction_samples = []
for i in range(min(20, len(test_inputs))):
    sample_text = texts[len(train_inputs) + len(val_inputs) + i]
    prediction_samples.append({
        'text': sample_text,
        'true_label': int(flat_true_labels[i]),
        'predicted_label': int(flat_predictions[i]),
        'true_sentiment': 'Positive' if flat_true_labels[i] == 1 else 'Negative',
        'predicted_sentiment': 'Positive' if flat_predictions[i] == 1 else 'Negative',
        'correct': flat_true_labels[i] == flat_predictions[i]
    })

with open(os.path.join(output_dir, 'prediction_samples.json'), 'w') as f:
    json.dump(prediction_samples, f, indent=2, ensure_ascii=False)
print("✅ Prediction samples saved to: prediction_samples.json")

# Create a summary report
summary = f"""
===========================================
BERT SENTIMENT ANALYSIS - FINAL REPORT
===========================================
MODEL: {config['model_name']}
DATASET: {config['data_path']}
TOTAL SAMPLES: {len(df)}

DATASET SPLIT:
  Training: {len(train_inputs)} samples
  Validation: {len(val_inputs)} samples
  Test: {len(test_inputs)} samples

CLASS DISTRIBUTION:
  Negative: {class_counts.get(0, 0)} samples
  Positive: {class_counts.get(1, 0)} samples

HYPERPARAMETERS:
  Epochs: {config['epochs']}
  Batch Size: {config['batch_size']}
  Learning Rate: {config['learning_rate']}
  Max Sequence Length: {config['max_len']}
  Dropout Rate: {config['dropout_rate']}
  Class Weights Used: {config['use_class_weights']}
  Early Stopping: {config['use_early_stopping']}

RESULTS:
  Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)
  Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
  Test Precision: {precision:.4f}
  Test Recall: {recall:.4f}
  Test F1 Score: {f1:.4f}

CONFUSION MATRIX:
  True Negatives: {cm[0,0]}
  False Positives: {cm[0,1]}
  False Negatives: {cm[1,0]}
  True Positives: {cm[1,1]}

TRAINING TIME: {format_time(time.time() - total_t0)}
===========================================
"""

with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
    f.write(summary)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(summary)

# Final verdict
print("\n" + "="*60)
if accuracy > 0.90:
    print("🎉🎉🎉 SUCCESS! ACCURACY ABOVE 90%! 🎉🎉🎉")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
else:
    print("⚠️  Accuracy below 90%. Consider these improvements:")
    print("   1. Increase training data quantity")
    print("   2. Try 'bert-large-multilingual-cased' model")
    print("   3. Add more data augmentation")
    print("   4. Use 5-fold cross-validation")
    print("   5. Try ensemble methods with multiple models")
    print("   6. Perform hyperparameter tuning with Optuna")
    print(f"   Current Accuracy: {accuracy*100:.2f}%")
print("="*60)

# Optional: Plot training history (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(stats_df['epoch'], stats_df['Training Loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(stats_df['epoch'], stats_df['Validation Loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(stats_df['epoch'], stats_df['Training Accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(stats_df['epoch'], stats_df['Validation Accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(stats_df['epoch'], stats_df['Validation F1'], 'g-', label='Validation F1', linewidth=2)
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Target', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📊 Training history plot saved as: training_history.png")
    
except ImportError:
    print("\nNote: Install matplotlib for training plots: pip install matplotlib")

print(f"\nAll files saved to directory: {output_dir}")
print("="*60)
print("SCRIPT EXECUTION COMPLETE!")
print("="*60)