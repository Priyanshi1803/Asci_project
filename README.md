# how to go about this repo
## coded scripts
the coded scripts contain all the scripts needed to be run to either fetch results like (top50_analysis.py, longest_Seq.py) or to generate new text you need to run test_withpen.py with 
the following command:
torchrun --nproc_per_node 1 /scratch/work/kharbap1/llama/coded_scripts/test_withpen.py     --ckpt_dir llama-2-7b/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6 --penalty 1.1 --output_file /scratch/work/kharbap1/generated_data/wiki_test_gen_1.1_128.txt
