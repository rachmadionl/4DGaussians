exp_name1=dnerf

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/standup/"  &
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/jumpingjacks/" &
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/bouncingballs/" &
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/lego/"   
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/hellwarrior/"  &
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/hook/" &
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/trex/" &
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/mutant/"   &
wait
echo "Done"
