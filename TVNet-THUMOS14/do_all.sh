
#Tensorflow

python TEM_train.py
python TEM_test.py

python VEM_create_windows.py --window_length 10 --window_stride 5
python VEM_create_windows.py --window_length 5 --window_stride 2

python VEM_train.py --voting_type start --window_length 10 --window_stride 5
python VEM_test.py --voting_type start --window_length 10 --window_stride 5

python VEM_train.py --voting_type end --window_length 10 --window_stride 5
python VEM_test.py --voting_type end --window_length 10 --window_stride 5

python VEM_train.py --voting_type start --window_length 5 --window_stride 2
python VEM_test.py --voting_type start --window_length 5 --window_stride 2

python VEM_train.py --voting_type end --window_length 5 --window_stride 2
python VEM_test.py --voting_type end --window_length 5 --window_stride 2


#Pytorch for PEM from BMN
python PEM_train.py
python proposal_generation.py
python post_postprocess.py

