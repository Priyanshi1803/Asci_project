import fire
from llama import Llama
from typing import List
import re

from llama import Llama
from typing import List
import re

def parse_wikitext(text):
    heading_pattern = re.compile(r'\s=\s(.*?)\s=\s')
    subtitle_pattern = re.compile(r'\s=\s=(.*?)=\s=\s')
    subtopic_pattern = re.compile(r'\s=\s=\s=(.*?)=\s=\s=\s')

    parsed_data = []
    current_heading = None
    current_subtitle = None
    current_subtopic = None

    for line in text.split('\n'):
        if len(line) <= 5:
             continue
        heading_match = heading_pattern.match(line)
        subtitle_match = subtitle_pattern.match(line)
        subtopic_match = subtopic_pattern.match(line)
        if subtopic_match:
            current_subtopic = subtopic_match.group(1).strip()
            parsed_data.append({'type': 'subtopic', 'text': current_subtopic})

        elif subtitle_match:
            current_subtitle = subtitle_match.group(1).strip()
            # print(current_subtitle)
            current_subtopic = None
            parsed_data.append({'type': 'subtitle', 'text': current_subtitle})
        elif heading_match:
            current_heading = heading_match.group(1).strip()
            
            current_subtitle = None
            current_subtopic = None
            parsed_data.append({'type': 'heading', 'text': current_heading})
        else:
            if current_heading:
                parsed_data.append({'type': 'content', 'text': line.strip(), 'heading': current_heading, 'subtitle': current_subtitle, 'subtopic': current_subtopic})

    return parsed_data


def main(
    ckpt_dir: str = '/scratch/work/kharbap1/llama/llama-2-7b',
    tokenizer_path: str= '/scratch/work/kharbap1/llama/tokenizer.model',
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 1028,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    output_file: str = '/scratch/work/kharbap1/asci/output.txt',
    input_file: str = '/scratch/work/kharbap1/wikitext-103-test.txt'  # New input file argument
):
    print(ckpt_dir)

    # Load and parse the input file
    with open(input_file, 'r') as f:
        text = f.read()

    # Debug: Print the raw text input
    # print("Raw Input Text:")
    # print(text[:500])  # Print first 500 characters for brevity

    parsed_data = parse_wikitext(text)

    # Debug: Print parsed data
    # print("Parsed Data:")
    # for entry in parsed_data:
    #     print(entry)

    # Generate prompts from the parsed data
    prompts = []
    for entry in parsed_data[:6]:
        if entry['type'] == 'content' and entry['heading']:
            prompt = f"{entry['heading']}\n\n{entry['text'].split('.')[0]}."
            if prompt.strip():  # Ensure the prompt is not empty
                prompts.append(prompt)

    if not prompts:
        raise ValueError("No valid prompts generated from the input data.")

    # Debug: Print the prompts
    print("Generated Prompts:")
    for prompt in prompts:
        print(prompt)

    # Initialize the Llama model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Generate text
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Print and save the results
    with open(output_file, 'w') as f:
        for prompt, result in zip(prompts, results):
            output_text = f"{prompt}\n> {result['generation']}\n\n==================================\n"
            print(output_text)
            f.write(output_text)

if __name__ == "__main__":
    fire.Fire(main)
