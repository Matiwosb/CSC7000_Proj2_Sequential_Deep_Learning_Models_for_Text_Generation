import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from text_model import (
    read_all_texts_from_folder,
    load_tokenizer,
    TextDataset,
    LSTMLanguageModel,
    RNNLanguageModel,
    TransformerLanguageModel,
    train_model,
    evaluate_model,
    generate_text
)
import matplotlib.pyplot as plt
import pandas as pd

def get_model(name, vocab_size, emb, hid, seq_len):
    """
    Utility function to initialize and return the selected model type.
    """
    if name == 'lstm':
        return LSTMLanguageModel(vocab_size, emb, hid)
    elif name == 'rnn':
        return RNNLanguageModel(vocab_size, emb, hid)
    elif name == 'transformer':
        return TransformerLanguageModel(vocab_size, emb, hid, max_seq_length=seq_len)
    else:
        raise ValueError("Unsupported model")

def main():
    """
    Main function to train and evaluate multiple language models (LSTM, RNN, Transformer).
    Produces loss curves, evaluation metric table, and model responses to predefined prompts.
    """
    model_types = ['lstm', 'rnn', 'transformer']
    results_table = []
    prompt1 = "Which do you prefer? Dogs or cats?"
    prompt2 = "Tell me a story about the stars."

    # ----------- Config ----------
    model_path = "tokenizer_bpe.model"
    vocab_size = 10000
    seq_len = 64
    batch_size = 128
    embedding_dim = 128
    hidden_dim = 128
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = load_tokenizer(model_path)

    # Load and tokenize dataset
    with open("all_text.txt", "r", encoding="utf-8") as f:
        all_text = f.read()

    dataset = TextDataset(all_text, tokenizer, seq_len=seq_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train and evaluate each model
    for model_type in model_types:
        print(f"\n\uD83D\uDE80 Training {model_type.upper()} model...")

        model = get_model(model_type, vocab_size, embedding_dim, hidden_dim, seq_len)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # Train model and get loss curves
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, optimizer, scheduler, criterion, device)

        # Save training/validation loss plot
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f"{model_type.upper()} Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{model_type}_loss_curve.png")
        plt.close()

        # Evaluate model
        prompts = [
            (prompt1, "I prefer cats."),
            ("What is the capital of France?", "Paris"),
            ("Who wrote Alice in Wonderland?", "Lewis Carroll")
        ]

        ppl, results = evaluate_model(model, tokenizer, val_loader, criterion, device, prompts)

        # Store results for comparison
        results_table.append({
            "Model": model_type.upper(),
            "Perplexity": round(ppl, 2),
            "Avg BLEU": round(sum(r['bleu'] for r in results) / len(results), 4),
            "Response to Prompt 1": generate_text(model, tokenizer, prompt1, max_len=50, temperature=1.0, device=device),
            "Response to Prompt 2": generate_text(model, tokenizer, prompt2, max_len=50, temperature=1.0, device=device),
        })

    # Display evaluation metrics in tabular format
    df = pd.DataFrame(results_table)
    print("\n\ud83d\udcca Final Evaluation Table:\n")
    print(df[["Model", "Perplexity", "Avg BLEU"]])

    # Print each model's response to selected prompts
    print("\n\ud83d\udd8e\ufe0f Prompt Responses:\n")
    for row in results_table:
        print(f"\nModel: {row['Model']}")
        print(f"\u2192 {prompt1}\n   {row['Response to Prompt 1']}")
        print(f"\u2192 {prompt2}\n   {row['Response to Prompt 2']}")

if __name__ == "__main__":
    main()
