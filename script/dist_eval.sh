python main.py \
--epoch 1 \
--batch_size 50 \
--classes_num 188 \
--model "Resnet" \
--data_dir "/data/dict/PE" \
--preprocessor "thumbnail" \
--dataset "raw" \
--checkpoint_dir "/data/dict/checkpoint_dist_0308_2/" \
--forward_only True
