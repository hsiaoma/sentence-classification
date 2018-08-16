python preprocess.py \
	--is_test=0 \
	--data_folder='data/' \
	--log_file='log/pre_process.log' \
	--mode='pad_and_concat' \
	--sent_len=20 \
	--list_len=10 \
    --num_workers=10 \
	--embedding_file='embedding_files/sgns.sogounews.bigram-char' \
    --words_file='embedding_files/words.txt' \
