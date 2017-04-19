python main.py \
--epoch 10000 \
--batch_size 32 \
--classes_num 394 \
--model "TDNN" \
--data_dir "/data/dict/PE/" \
--preprocessor "simhash" \
--dataset "raw" \
--checkpoint_dir "/data/dict/refactoring/checkpoint_simhash_0329/" \
--forward_only True 
