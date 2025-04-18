import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import math

# ------------------------
# 1. Tokenizer Training and Loading
# ------------------------
def train_tokenizer(input_path, vocab_size=10000):
    """
    Train a SentencePiece tokenizer on a given text file.
    """
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix='tokenizer_bpe',
        vocab_size=vocab_size,
        model_type='bpe',
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )

def load_tokenizer(model_path='tokenizer_bpe.model'):
    """
    Load a trained SentencePiece tokenizer.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

# ------------------------
# 2. Dataset
# ------------------------
def read_all_texts_from_folder(folder_path):
    """
    Read all .txt files from a directory and combine them into one string.
    """
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return '\n'.join(texts)

class TextDataset(Dataset):
    """
    Custom dataset to return token sequences and their corresponding targets.
    """
    def __init__(self, data_text, tokenizer, seq_len=128):
        self.data = tokenizer.encode(data_text, out_type=int)
        self.seq_len = seq_len
        self.samples = [self.data[i:i+seq_len+1] for i in range(len(self.data)-seq_len)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.samples[idx][1:], dtype=torch.long)
        return x, y

# ------------------------
# 3. Base Model
# ------------------------
class BaseLanguageModel(nn.Module):
    """
    Abstract base class for language models.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(BaseLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        raise NotImplementedError()

    def sample_next_token(self, logits, temperature=1.0):
        """
        Sample the next token given logits and temperature.
        """
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

# ------------------------
# 4. RNN Model
# ------------------------
class RNNLanguageModel(BaseLanguageModel):
    """
    Simple RNN-based language model.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc_out(out)
        return logits, hidden

# ------------------------
# 5. LSTM Model
# ------------------------
class LSTMLanguageModel(BaseLanguageModel):
    """
    LSTM-based language model.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc_out(out)
        return logits, hidden

# ------------------------
# 6. Transformer Model
# ------------------------
class TransformerLanguageModel(BaseLanguageModel):
    """
    Transformer encoder-based language model.
    """
    def __init__(self, vocab_size, embedding_dim, nhead=4, num_layers=2, max_seq_length=512):
        super().__init__(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, hidden=None):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = x.permute(1, 0, 2)  # (seq_len, batch, emb)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, emb)
        logits = self.fc_out(x)
        return logits, None

# ------------------------
# 7. Text Generation
# ------------------------
def generate_text(model, tokenizer, prompt, max_len=50, temperature=0.7, device="cuda"):
    """
    Generate text autoregressively from a prompt.
    """
    model.eval()
    tokens = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    output = input_ids
    hidden = None
    eos_token = tokenizer.eos_id() if callable(tokenizer.eos_id) else tokenizer.eos_id

    for _ in range(max_len):
        with torch.no_grad():
            if isinstance(model, TransformerLanguageModel):
                logits, _ = model(output)
                next_token_logits = logits[:, -1, :]
            else:
                logits, hidden = model(output, hidden)
                next_token_logits = logits[:, -1, :]

            next_token = model.sample_next_token(next_token_logits, temperature)
            next_token = next_token.unsqueeze(1)  # (1, 1)
            output = torch.cat([output, next_token], dim=1)

            if next_token.item() == eos_token:
                break

    return tokenizer.decode(output[0].tolist())

# ------------------------
# 8. Training Loop
# ------------------------
def train_model(model, train_loader, val_loader, epochs, optimizer, scheduler, criterion, device):
    """
    Train the language model with early stopping.
    Returns training and validation loss curves.
    """
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    return train_losses, val_losses

# ------------------------
# 9. Evaluation Metrics
# ------------------------
def compute_perplexity(loss):
    """Convert average loss to perplexity."""
    return math.exp(loss)

def compute_bleu(reference, generated):
    """Compute BLEU score with smoothing."""
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)

def evaluate_model(model, tokenizer, data_loader, criterion, device, prompt_list):
    """
    Evaluate the model with perplexity and BLEU using custom prompts.
    """
    model.to(device)
    model.eval()

    # Compute loss-based perplexity
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    ppl = compute_perplexity(avg_loss)

    # Evaluate generation
    results = []
    for prompt, reference in prompt_list:
        generated = generate_text(model, tokenizer, prompt, max_len=50, temperature=1.0, device=device)
        bleu = compute_bleu(reference, generated)
        
        results.append({
            "prompt": prompt,
            "reference": reference,
            "generated": generated,
            "bleu": bleu
      
        })

    return ppl, results
