# CSC7000 Foundational AI Project 2: Sequential Deep Learning Models for Text Generation
# 📚 Language Model Training and Evaluation

This project trains and evaluates three types of language models using PyTorch:
- RNN
- LSTM
- Transformer

It uses a SentencePiece tokenizer and provides evaluation via **perplexity** and **BLEU** scores, along with **text generation**.

---

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Matiwosb/CSC7000_Proj2_Sequential_Deep_Learning_Models_for_Text_Generation/
cd CSC7000_Proj2_Sequential_Deep_Learning_Models_for_Text_Generation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK resources**:
```python
# In a Python shell or script:
import nltk
nltk.download('punkt')
```

---

## 🧪 How to Run

Train and evaluate all models:

```bash
python main.py
```

This script will:
- Load your tokenizer and dataset from `all_text.txt`
- Train each model (LSTM, RNN, Transformer)
- Plot and save training/validation loss curves
- Evaluate each model on sample prompts
- Print a final table with metrics and generation samples

---

## 📊 Outputs

1. **Loss Plots**
   - `lstm_loss_curve.png`
   - `rnn_loss_curve.png`
   - `transformer_loss_curve.png`

2. **Console Outputs**
   - Evaluation table with **Perplexity** and **BLEU**
   - Model responses to:
     - `"Which do you prefer? Dogs or cats?"`
     - `"Tell me a story about the stars."`

---

## ✏️ Customization

### ➔ Add Your Own Prompts
Modify the `prompts` list in `test_main.py`:
```python
prompt1 = "Your new question here?"
prompt2 = "Another custom prompt."
```

### ➔ Change Model Settings
Inside `test_main.py`, update:
```python
embedding_dim = 256
hidden_dim = 512
epochs = 20
```

---

## 📦 Training Your Tokenizer

You can train a new tokenizer using:
```python
from txt_model import train_tokenizer
train_tokenizer("your_data.txt")
```

---

## 📄 License

MIT License — Feel free to reuse and modify for educational or research purposes.

