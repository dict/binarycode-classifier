python main.py \
--epoch 1 \
--batch_size 50 \
--classes_num 188 \
--model "Resnet" \
--data_dir "/data/dict/PE" \
--preprocessor "thumbnail" \
--dataset "raw" \
--checkpoint_dir "/data/dict/refactoring/checkpoint_dist/" \
--forward_only True
