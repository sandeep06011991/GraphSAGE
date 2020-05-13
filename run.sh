export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
python experiment/epoch_run_time.py ./example_data/toy-ppi
python experiment/epoch_run_time.py ./example_data/reddit/reddit
python experiment/epoch_run_time.py ./example_data/ppi/ppi
python -m graphsage.supervised_train --train_prefix ./example_data/reddit/reddit --model graphsage_maxpool --sigmoid
python -m graphsage.supervised_train --train_prefix ./example_data/ppi/ppi --model graphsage_maxpool --sigmoid
python -m graphsage.unsupervised_train --train_prefix ./example_data/reddit/reddit --model gcn --max_total_steps 1000 --validate_iter 10
python -m graphsage.unsupervised_train --train_prefix ./example_data/ppi/ppi --model gcn --max_total_steps 1000 --validate_iter 10

