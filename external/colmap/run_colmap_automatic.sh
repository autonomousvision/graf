#set -e

input_dir=$1
output_dir=$2

mkdir -p ${output_dir}
echo Processing ${input_dir} ...
colmap automatic_reconstructor \
  --workspace_path ${output_dir} \
  --image_path  ${input_dir}/ \
  --single_camera=1 \
  --dense=1 \
  --gpu_index=0
