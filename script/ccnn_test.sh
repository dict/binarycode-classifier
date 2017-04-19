python main.py \
--epoch 1 \
--batch_size 2 \
--classes_num 200 \
--model "TDNN" \
--data_dir "/data/dict/PE/" \
--preprocessor "sequence" \
--dataset "raw" \
--checkpoint_dir "/data/dict/refactoring/checkpoint_ccnn_0328/" \
--forward_only True
