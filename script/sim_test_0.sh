python main.py \
--epoch 1 \
--batch_size 32 \
--classes_num 394 \
--model "TDNN" \
--data_dir "/data/dict/PE/" \
--preprocessor "simhash" \
--dataset "raw" \
--GPU "5" \
--binary_embed_width 32 \
--binary_embed_height 24000 \
--checkpoint_dir "/data/dict/refactoring/checkpoint_simhash_0330/0/" \
--forward_only True
