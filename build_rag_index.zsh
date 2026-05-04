python ./src/index_builder.py \
  --input-dir $LLAMA_PREPARED_DIR \
  --output-dir $LLAMA_RAG_DIR \
  --chunk-size-chars 1800 \
  --overlap-chars 250 \
  --batch-size 32
