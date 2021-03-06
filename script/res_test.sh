python main.py \
--epoch 1 \
--batch_size 50 \
--classes_num 1001 \
--model "Resnet" \
--data_dir "/data/dict/PE/" \
--data_type "db" \
--preprocessor "thumbnail" \
--dataset "raw" \
--checkpoint_dir "/data/dict/refactoring/checkpoint_resnet_0322_white/" \
--forward_only True 
