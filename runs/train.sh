export CUDA_VISIBLE_DEVICES=0,1

CONFIG_PATH=configs/default.json
VERBOSE="yes"

python train.py -c $CONFIG_PATH -v $VERBOSE
