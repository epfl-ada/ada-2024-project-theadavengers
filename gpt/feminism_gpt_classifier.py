from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2Model
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import torch.nn as nn


class GPTClassifier(nn.Module):
        def __init__(self, gpt_model, num_classes):
            super(GPTClassifier, self).__init__()
            self.gpt = gpt_model
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.gpt.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, -1, :]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_feminism_classifier(data, summaries_col='Summaries', label_col='feminism', num_epochs=3, batch_size=8, lr=5e-5, name = 'best_model_state.bin'):
    """
    Train a classifier to predict feminism indicators from summaries using GPT-2 embeddings.
    
    Args:
    - data (pd.DataFrame): Input DataFrame with summaries and labels.
    - summaries_col (str): Name of the column containing summaries (default: 'Summaries').
    - label_col (str): Name of the column containing labels (default: 'feminism').
    - num_epochs (int): Number of training epochs (default: 3).
    - batch_size (int): Batch size for training (default: 8).
    - lr (float): Learning rate for optimizer (default: 5e-5).
    
    Returns:
    - model (torch.nn.Module): The trained model.
    - metrics (dict): Training and testing metrics (loss, accuracy) per epoch.
    """

    # Prepare Dataset
    data_dict = {
        'Summaries': data[summaries_col].tolist(),
        'feminism': data[label_col].tolist()
    }
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch[summaries_col], padding=True, truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', label_col])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', label_col])

    # Define Model
    gpt_model = GPT2Model.from_pretrained('gpt2')

    class GPTClassifier(nn.Module):
        def __init__(self, gpt_model, num_classes):
            super(GPTClassifier, self).__init__()
            self.gpt = gpt_model
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.gpt.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, -1, :]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

    model = GPTClassifier(gpt_model, num_classes=2)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, correct_predictions = 0, 0
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch[label_col].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

        return total_loss / len(loader), correct_predictions.double() / len(loader.dataset)

    def eval_model(model, loader, criterion):
        model.eval()
        total_loss, correct_predictions = 0, 0
        with torch.no_grad():
            for batch in tqdm(loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch[label_col].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)

        return total_loss / len(loader), correct_predictions.double() / len(loader.dataset)

    metrics = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    best_accuracy = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = eval_model(model, test_loader, criterion)

        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), name)

    return model, metrics
