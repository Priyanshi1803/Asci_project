# how to go about this repo
## coded scripts
1) the coded scripts contain all the scripts needed to be run to either fetch results like (top50_analysis.py, longest_Seq.py)
2) to generate new text you need to run test_withpen.py with 
the following command:
torchrun --nproc_per_node 1 /scratch/work/kharbap1/llama/coded_scripts/test_withpen.py     --ckpt_dir llama-2-7b/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6 --penalty 1.1 --output_file /scratch/work/kharbap1/generated_data/wiki_test_gen_1.1_128.txt

###llama
my_generation.py is the script called while generation, edit has been made by adding penalty implemented from CTRL paper, with few changes like penalty window and handling negative & positive logits.

####generated data
1)this folder has all the datasets which have been generated, the penalty used and the window size has been appended in the name of the file itself (wiki_test_gen_1.3_256)
given 512 tokens iteratively to generate the next 512 tokens.

2)results file contain analysis of avg non broken sentences, longest repeating subsequences etc. 
