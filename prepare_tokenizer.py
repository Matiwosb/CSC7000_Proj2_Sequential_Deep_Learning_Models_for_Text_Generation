from text_model import read_all_texts_from_folder, train_tokenizer

# Step 1: Read all text files from the raw data folder
folder_path = "/Users/matiwosbirbo/CSC7700_Fondational_AI/Project2/UnderWork/raw_data"
all_text = read_all_texts_from_folder(folder_path)

# Step 2: Save the combined text into a single file
with open("all_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

# Step 3: Train the tokenizer using SentencePiece on the combined text
train_tokenizer("all_text.txt")

print("✅ Tokenizer training complete. Files 'tokenizer_bpe.model' and 'tokenizer_bpe.vocab' created.")
