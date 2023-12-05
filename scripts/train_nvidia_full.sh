exp_name=nvidia_dvs_full
config_name=hypernerf
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/nvidia_full/Balloon1/dense --port 6068 --expname "$exp_name/Balloon1" --configs arguments/$config_name/default.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/$exp_name/Balloon1 --configs arguments/$config_name/default.py --skip_train
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name/Balloon1/"
echo "Done"