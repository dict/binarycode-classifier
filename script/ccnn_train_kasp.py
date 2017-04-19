python main.py \
--epoch 10000 \
--batch_size 16 \
--classes_num 414 \
--model "TDNN" \
--data_dir "/data/dict/PE/" \
--preprocessor "sequence" \
--data_type "db" \
--dropout_prob 0.5 \
--learning_rate 0.0001 \
--dataset "raw" \
--GPU "2,3" \
--feature_maps "[100,200,300,400,500]" \
--kernels "[2,4,6,8,10]" \
--binary_embed_width 32 \
--binary_embed_height 36000 \
--checkpoint_dir "/data/dict/refactoring/checkpoint_ccnn_kasp_0407/0/" \
--forward_only False
