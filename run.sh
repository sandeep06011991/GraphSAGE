export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
#rm  -f experiments.txt
#python experiment/epoch_run_time.py ./example_data/toy-ppi
#python experiment/epoch_run_time.py ./example_data/reddit/reddit
#python experiment/epoch_run_time.py ./example_data/ppi/ppi
 
echo "dataset | gpu_avail | constant/Feed | is_projected(or Actual)|  time \n" >> tf_measurements.txt
python experiment/tensorflow_sampling.py ../datasets/cit-Patents.txt gpu 240
python experiment/tensorflow_sampling.py ../datasets/cit-Patents.txt cpu 240




#python experiment/data_post_process.py ./example_data/reddit/reddit && cp edgelist reddit_edgelist
#python experiment/data_post_process.py ./example_data/ppi/ppi && cp edgelist ppi_edgelist
#python -m graphsage.supervised_train --train_prefix ./example_data/reddit/reddit --model graphsage_maxpool --sigmoid
#python -m graphsage.supervised_train --train_prefix ./example_data/ppi/ppi --model graphsage_maxpool --sigmoid
#python -m graphsage.unsupervised_train --train_prefix ./example_data/reddit/reddit --model gcn --max_total_steps 1000 --validate_iter 10
#python -m graphsage.unsupervised_train --train_prefix ./example_data/ppi/ppi --model gcn --max_total_steps 1000 --validate_iter 10

