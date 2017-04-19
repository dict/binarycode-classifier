python main.py \
--epoch 100 \
--batch_size 32 \
--classes_num 200 \
--model "TDNN" \
--data_dir "/data/dict/PE/" \
--preprocessor "sequence" \
--dataset "raw" \
--checkpoint_dir "/data/dict/refactoring/checkpoint_ccnn_0328/" \
--forward_only False
