python main.py \
--epoch 10000 \
--batch_size 16 \
--classes_num 394 \
--model "TDNN" \
--data_dir "/data/files/vato_clone/VIRUSCRAWLER/storage/Win32 EXE" \
--preprocessor "sequence" \
--label_query "select * from pe_list where num > 9" \
--data_query "select * from win32_train_list_clone order by md5" \
--data_type "db" \
--dropout_prob 0.5 \
--learning_rate 0.0001 \
--dataset "raw" \
--GPU "0,1" \
--feature_maps "[100,200,300,400,500]" \
--kernels "[2,4,6,8,10]" \
--binary_embed_width 32 \
--binary_embed_height 36000 \
--checkpoint_dir "/data/dict/refactoring/checkpoint_ccnn_0408/0/" \
--forward_only False
