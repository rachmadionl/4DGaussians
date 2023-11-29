while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--data_dir) dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "DATA DIR: $dir"
DATASET_PATH=$dir
colmap_db_path=$DATASET_PATH/database.db
image_path=$DATASET_PATH/images_colmap
output_path=$DATASET_PATH/sparse
mkdir $output_path

colmap feature_extractor \
--SiftExtraction.use_gpu 0 \
--SiftExtraction.upright 1 \
--ImageReader.camera_model OPENCV \
--ImageReader.single_camera 1 \
--ImageReader.mask_path $DATASET_PATH/background_mask \
--database_path $colmap_db_path \
--image_path $image_path

colmap exhaustive_matcher \
    --SiftMatching.use_gpu 0 \
    --database_path $colmap_db_path

min_num_matches=32
filter_max_reproj_error=2
tri_complete_max_reproj_error=2

colmap mapper \
  --Mapper.ba_refine_principal_point 1 \
  --Mapper.filter_max_reproj_error $filter_max_reproj_error \
  --Mapper.tri_complete_max_reproj_error $tri_complete_max_reproj_error \
  --Mapper.min_num_matches $min_num_matches \
  --database_path $colmap_db_path \
  --image_path $image_path \
  --output_path $output_path

# colmap feature_extractor \
# --database_path $colmap_db_path \
# --image_path $image_path \
# --SiftExtraction.upright 1 \
# --ImageReader.camera_model OPENCV \
# --ImageReader.mask_path $DATASET_PATH/background_mask \
# --ImageReader.single_camera 1

# colmap exhaustive_matcher \
# --database_path $colmap_db_path

# mkdir $DATASET_PATH/sparse
# colmap mapper \
#     --database_path $colmap_db_path \
#     --image_path $image_path \
#     --output_path $output_path \
#     --Mapper.num_threads 16 \
#     --Mapper.init_min_tri_angle 4 \
#     --Mapper.multiple_models 0 \
#     --Mapper.extract_colors 0