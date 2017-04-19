python main.py \
--epoch 10000 \
--batch_size 16 \
--classes_num 394 \
--model "TDNN" \
--data_dir "/data/dict/PE/" \
--preprocessor "sequence" \
--data_type "db" \
--dataset "raw" \
--GPU "0,1,2,3" \
--feature_maps "[100,200,300,400,500]" \
--kernels "[2,4,6,8,10]" \
--binary_embed_width 32 \
--binary_embed_height 32000 \
--checkpoint_dir "/data/dict/refactoring/checkpoint_ccnn_0403/3/" \
--forward_only True
