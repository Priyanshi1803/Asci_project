#this file is basically used to take all the result generations and calculate the longest repeating
# subsequence and also check if the sentence is broken, avg nonbroken sequence length in text
import re
from collections import defaultdict

def find_repeated_substrings(text):
    def find_substrings(words, length):
        substrings = defaultdict(int)
        for i in range(len(words) - length + 1):
            substr = ' '.join(words[i:i + length])
            substrings[substr] += 1
        return substrings

   
    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()
    words = cleaned_text.split()

   
    min_length = 2  
    all_substrings = defaultdict(int)
    for length in range(min_length, len(words) // 2 + 1):
        substrings = find_substrings(words, length)
        for substr, count in substrings.items():
            if count > 1:
                all_substrings[substr] = count

    
    max_length = 0
    longest_substrings = {}
    for substr, count in all_substrings.items():
        length = len(substr.split())
        if length > max_length:
            max_length = length
            longest_substrings = {substr: (length, count)}
        elif length == max_length:
            longest_substrings[substr] = (length, count)

    return longest_substrings

def is_broken(sentence):
    words = sentence.split()
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True
    return False

def average_sentence_length(text):
    
    sentences = re.split(r'[.!?]\s*', text)
  
    sentences = [sentence for sentence in sentences if sentence]

    non_broken_sentence_lengths = []
    broken_sentence_lengths = []
    broken_sentences = []

    for sentence in sentences:
        words = sentence.split()
        length = len(words)
        
        if is_broken(sentence):
            broken_sentence_lengths.append(length)
            broken_sentences.append(sentence)
        else:
            non_broken_sentence_lengths.append(length)

   
    if non_broken_sentence_lengths:
        average_non_broken_length = sum(non_broken_sentence_lengths) / len(non_broken_sentence_lengths)
    else:
        average_non_broken_length = 0
    
    if broken_sentence_lengths:
        average_broken_length = sum(broken_sentence_lengths) / len(broken_sentence_lengths)
    else:
        average_broken_length = 0

    return average_non_broken_length, average_broken_length, broken_sentences


def parse_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    parsed_results = {}
    current_key = None

    for line in lines:
        line = line.strip()
        if line.isdigit():
            current_key = int(line)
            parsed_results[current_key] = []
        elif current_key is not None:
            parsed_results[current_key].append(line)

  
    for key in parsed_results:
        parsed_results[key] = ''.join(parsed_results[key])
        
  
    with open(output_filename, 'w') as output_file:
        for key, content in parsed_results.items():
            output_file.write(f"{key}:\n{content}\n\n")

            output_file.write("Repeated Substrings:\n")
            repeated_substrings = find_repeated_substrings(content)
            for substr, (length, count) in repeated_substrings.items():
                output_file.write(f"'{substr}' (Length: {length}, Count: {count})\n\n")

            output_file.write("avg distances:\n")
            avg_non_broken_length, avg_broken_length, broken_sentences = average_sentence_length(content)
            output_file.write(f"Average non-broken sentence length: {avg_non_broken_length}\n\n")
            output_file.write(f"Average broken sentence length: {avg_broken_length}\n\n")

            output_file.write("Broken Sentences Detected:\n")
            for sentence in broken_sentences:
                output_file.write(f"{sentence}\n")


input_filename = '/scratch/work/kharbap1/generated_data/wiki_test_gen_1.3_256'
output_filename = '/scratch/work/kharbap1/generated_data/results_1.3_256.txt'
parse_file(input_filename, output_filename)
