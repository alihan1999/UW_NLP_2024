#!/bin/bash

modes=("train" "dev" "test")
models=("unigram" "bigram" "trigram")

for mode in "${modes[@]}"; do
for model in "${models[@]}"; do
python3 ngram.py --model "$model" --dataset "$mode"
python3 ngram.py --model "$model" --dataset "$mode" -s
done
done