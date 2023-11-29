# Generate json data
# NVIDIA DATASET

data=Balloon1-2
data_dir=data/nvidia_full/$data
python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir
python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/mv_images --data_dir $data_dir

# data=Balloon2-2
# data_dir=data/nvidia_full/$data
# python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
# bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
# python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir

# data=DynamicFace-2
# data_dir=data/nvidia_full/$data
# python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
# bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
# python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir

# data=Jumping
# data_dir=data/nvidia_full/$data
# python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
# bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
# python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir

# data=Playground
# data_dir=data/nvidia_full/$data
# python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
# bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
# python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir

# data=Skating-2
# data_dir=data/nvidia_full/$data
# python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
# bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
# python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir

# data=Truck-2
# data_dir=data/nvidia_full/$data
# python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
# bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
# python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir

# data=Umbrella
# data_dir=data/nvidia_full/$data
# python utils/data_preprocessing/generate_data.py --path /home/rachmadio/dev/data/dynamic_scene_data_full/nvidia_data_full/$data/dense/images --data_dir $data_dir
# bash utils/data_preprocessing/register_camera.sh --data_dir $data_dir
# python utils/data_preprocessing/save_camera_data.py --data_dir $data_dir

echo "D O N E !"