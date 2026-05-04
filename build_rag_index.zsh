python ./src/index_builder.py \
  --input-dir /datasets/llama/text \
  --output-dir ~/llrun/gutenberg_bge_index \
  --chunk-size-chars 1800 \
  --overlap-chars 250 \
  --batch-size 32
