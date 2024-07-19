import os
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from logging import getLogger
from sentencepiece import SentencePieceProcessor
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

logger = getLogger()

class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
   
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)


def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_table_to_file(table_data: List[List[str]], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("Top 50 Frequent Tokens\n")
        file.write("-" * 50 + "\n")
        file.write("{:<10} | {:<20} | {}\n".format("Encoded ID", "Decoded Token", "Frequency"))
        file.write("-" * 50 + "\n")
        for row in table_data:
            file.write("{:<10} | {:<20} | {}\n".format(row[0], row[1], row[2]))
            
def tokenize_and_count(text: str, tokenizer: Tokenizer) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, float]]:
    all_tokens = tokenizer.encode(text, bos=False, eos=False)
    token_frequency_data = defaultdict(int)
    token_indices_data = defaultdict(list)
    token_avg_distance_data = {}

    for index, token in enumerate(all_tokens):
        token_frequency_data[token] += 1
        token_indices_data[token].append(index)

    for token, positions in token_indices_data.items():
        if len(positions) > 1:
            distances = [positions[i] - positions[i - 1] for i in range(1, len(positions))]
            avg_distance = sum(distances) / len(distances)
            token_avg_distance_data[token] = avg_distance
        else:
            token_avg_distance_data[token] = 0  # Average distance is 0 since the token occurs once

    return token_frequency_data, token_indices_data, token_avg_distance_data

def main(input_file: str, model_path: str, output_file: str):
    tokenizer = Tokenizer(model_path)
    text = read_file(input_file)
    
    
    token_frequency_data, token_indices_data, token_avg_distance_data = tokenize_and_count(text, tokenizer)
    
    
    filtered_data = []
    for token, frequency in token_frequency_data.items():
        decoded_token = tokenizer.decode([token]).strip().lower()
        if decoded_token not in stop_words and decoded_token.isalpha() and len(decoded_token) > 1:
            avg_distance = token_avg_distance_data[token]
            indices = token_indices_data[token]
            filtered_data.append((token, decoded_token, frequency, avg_distance, indices))

    top_50_by_frequency = sorted(filtered_data, key=lambda x: x[2], reverse=True)[:50]
    
   
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("Top 50 Frequent Tokens\n")
        file.write("-" * 70 + "\n")
        file.write("{:<10} | {:<20} | {:<10} | {:<10} | {}\n".format("Encoded ID", "Decoded Token", "Frequency", "Avg Dist", "Indices"))
        file.write("-" * 70 + "\n")
        for row in top_50_by_frequency:
            file.write("{:<10} | {:<20} | {:<10} | {:<10.2f} | {}\n".format(row[0], row[1], row[2], row[3], row[4]))
        
    
if __name__ == "__main__":

    input_file = "/scratch/work/kharbap1/generated_data/wiki_Test_generated.txt"
    model_path = "/scratch/work/kharbap1/files/wiki_senpiece.model"
    output_file = "/scratch/work/kharbap1/results/top50_wiki_generated"
    main(input_file, model_path, output_file)



