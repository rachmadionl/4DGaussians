exp_name1=dnerf_new
config_name=dnerf

# TRAIN
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/lego --port 6068 --expname "$exp_name1/lego" --configs arguments/$config_name/lego.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/bouncingballs --port 6066 --expname "$exp_name1/bouncingballs" --configs arguments/$config_name/bouncingballs.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/jumpingjacks --port 6069 --expname "$exp_name1/jumpingjacks" --configs arguments/$config_name/jumpingjacks.py  &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/trex --port 6070 --expname "$exp_name1/trex" --configs arguments/$config_name/trex.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/mutant --port 6068 --expname "$exp_name1/mutant" --configs arguments/$config_name/mutant.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/standup --port 6066 --expname "$exp_name1/standup" --configs arguments/$config_name/standup.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/hook --port 6069 --expname "$exp_name1/hook" --configs arguments/$config_name/hook.py  &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/hellwarrior --port 6070 --expname "$exp_name1/hellwarrior" --configs arguments/$config_name/hellwarrior.py
wait
# RENDER
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/standup/"  --skip_train --configs arguments/$config_name/standup.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/jumpingjacks/"  --skip_train --configs arguments/$config_name/jumpingjacks.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/bouncingballs/"  --skip_train --configs arguments/$config_name/bouncingballs.py  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/lego/"  --skip_train --configs arguments/$config_name/lego.py  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/hellwarrior/"  --skip_train --configs arguments/$config_name/hellwarrior.py  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/hook/"  --skip_train --configs arguments/$config_name/hook.py  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/trex/"  --skip_train --configs arguments/$config_name/trex.py  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/$exp_name1/mutant/"  --skip_train --configs arguments/$config_name/mutant.py   &
wait
# METRICS
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
echo "Done"
