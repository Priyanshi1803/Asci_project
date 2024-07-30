# LLaMA Text Generation and Analysis

This repository contains scripts and data for generating text using the LLaMA model with various penalties and analyzing the generated text.

## Repository Structure

### Coded Scripts
This directory contains all the necessary scripts to run and fetch results:

1. `top50_analysis.py` - Script to analyze the top 50 generated sequences.
2. `longest_Seq.py` - Script to find the longest repeating subsequences in the generated text.
3. `test_withpen.py` - Script to generate new text with specified penalties.

#### Running Text Generation

To generate new text, run the `test_withpen.py` script with the following command:

```sh
torchrun --nproc_per_node 1 /scratch/work/kharbap1/llama/coded_scripts/test_withpen.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6 \
    --penalty 1.1 \
    --output_file /scratch/work/kharbap1/generated_data/wiki_test_gen_1.1_128.txt

