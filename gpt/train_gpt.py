import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2Model, AdamW
from tqdm import tqdm

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


class GPT2FeminismClassifier:
    def __init__(self, train_data, test_data, tokenizer, model_name='gpt2', num_classes=2, batch_size=8, lr=5e-5, epochs=3):
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        
        # Initialize the GPT2 model
        gpt_model = GPT2Model.from_pretrained(self.model_name)
        self.model = GPTClassifier(gpt_model, self.num_classes)
        
        # Setup optimizer and loss function
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Prepare data loaders
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size)

    def tokenize(self, batch):
        return self.tokenizer(batch['Summaries'], padding=True, truncation=True, max_length=128)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['feminism'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

        return total_loss / len(loader), correct_predictions.double() / len(loader.dataset)

    def eval_model(self, loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch in tqdm(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['feminism'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
        
        return total_loss / len(loader), correct_predictions.double() / len(loader.dataset)

    def fit(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss, train_acc = self.train_epoch(self.train_loader)
            test_loss, test_acc = self.eval_model(self.test_loader)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), 'best_model_state.bin')

    def predict(self, summaries):
        # Prepare the input for prediction
        inputs = self.tokenizer(summaries, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            preds = torch.argmax(logits, dim=1)
        return preds

# Tokenizer setup
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT2 does not have a pad token by default

# Prepare the dataset
data_dict = {
    'Summaries': feminism_indicator['Summaries'].tolist(),
    'feminism': feminism_indicator['feminism'].tolist()
}
dataset = Dataset.from_dict(data_dict)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
test_dataset = dataset['test']

# Tokenize the dataset
train_dataset = train_dataset.map(lambda x: tokenizer(x['Summaries'], padding=True, truncation=True, max_length=128), batched=True)
test_dataset = test_dataset.map(lambda x: tokenizer(x['Summaries'], padding=True, truncation=True, max_length=128), batched=True)

# Convert to torch tensors
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'feminism'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'feminism'])

# Initialize the pipeline
classifier = GPT2FeminismClassifier(train_dataset, test_dataset, tokenizer)

# Train the model
classifier.fit()

# Example prediction (for a single or batch of summaries)
summaries_to_predict = ["This is a sample summary about feminism.", "Another example summary."]
predictions = classifier.predict(summaries_to_predict)
print(predictions)
