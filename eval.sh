export CUDA_VISIBLE_DEVICES=3,4
torchrun --nproc-per-node=2 train_wav2vec_bert.py eval_wav2vec_bert.json