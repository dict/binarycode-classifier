python main.py \
--epoch 10000 \
--batch_size 32 \
--classes_num 394 \
--model "TDNN" \
--data_dir "/data/dict/PE/" \
--preprocessor "simhash" \
--dataset "raw" \
--GPU "0,1,2,3" \
--binary_embed_width 32 \
--binary_embed_height 12000 \
--checkpoint_dir "/data/dict/refactoring/checkpoint_simhash_0330/" \
--forward_only False
