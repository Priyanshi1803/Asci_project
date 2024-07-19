
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from tqdm import tqdm
from llama import Llama
from typing import List
def main(
    ckpt_dir: str = '/scratch/work/kharbap1/llama/llama-2-7b',
    tokenizer_path: str= '/scratch/work/kharbap1/llama/tokenizer.model',
    temperature: float = 0.7, #0.6
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 512,
    max_batch_size: int = 6,
    penalty: float=1.3,
    output_file: str = '/scratch/work/kharbap1/onearticle_withpen'
):
    print(ckpt_dir)
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    with open('/scratch/work/kharbap1/files/wikitext-103-test.txt', 'r') as f:
        text = f.read()
        
    token_chunks = [text[i:i+512] for i in range(0, len(text),512)]
    
    results=[]
    p=[]
    for chunk in tqdm(token_chunks, desc="Processing chunks"):
    # for chunk in token_chunks:
        prompts = [chunk]
        p.append(chunk)
        # Generate the next 512 tokens for the current chunk
        
        chunk_results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            penalty=penalty
        )

        results.extend(chunk_results)

    
    
    # for prompt, result in zip(prompts, results):
    #      print(prompt)
    #      print(f"> {result['generation']}")
    #      print("\n==================================\n")

    # with open(output_file, 'w') as f:
    #     for prompt, result in zip(p, results):
    #         output_text = f"{prompt}\n> {result['generation']}\n\n==================================\n"
    #         # print(output_text)
    #         f.write(output_text)
    with open(output_file, 'w') as f:
        for idx, result in enumerate(results, start=1):
            output_text = f"{idx}\n> {result['generation']}\n\n"
            f.write(output_text)

if __name__ == "__main__":
    fire.Fire(main)
    print('task done')


