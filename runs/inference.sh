CONFIG_PATH=$1  #configs/default.json
CHECKPOINT_PATH=$2  #logs/pretrained_ljspeech.pt
NOISE_SCHEDULE_PATH=$3  #schedules/default/6iters.pt
MEL_FILELIST_PATH=$4  #tmp/mel_filelist.txt
VERBOSE=$5   #'yes'

python inference.py -c $CONFIG_PATH -ch $CHECKPOINT_PATH -ns $NOISE_SCHEDULE_PATH -m $MEL_FILELIST_PATH -v $VERBOSE
