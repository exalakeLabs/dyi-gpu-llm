python ./src/index_builder.py \
  --input-dir $LLAMA_PREPARED_DIR \
  --output-dir $LLAMA_RAG_DIR \
  --embed-model BAAI/bge-m3 \
  --chunk-size-chars 1800 \
  --overlap-chars 250 \
  --batch-size 32
