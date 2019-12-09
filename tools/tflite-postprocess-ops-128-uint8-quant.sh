#!/usr/bin/env bash

TMP_DIR=${HOME}/tmp
TF_MODELS_DIR=${HOME}/projects/models
TF_MODELS_PY=${HOME}/projects/models/.venv/bin/python
MODEL_NAME=ssd_mobilenet_v3_small_coco_2019_08_14
MODEL_URL=http://download.tensorflow.org/models/object_detection/${MODEL_NAME}.tar.gz
MODEL_DIR=${TMP_DIR}/${MODEL_NAME}
CONFIG_FILE=${MODEL_DIR}/pipeline.config
CHECKPOINT=${MODEL_DIR}/model.ckpt

curl ${MODEL_URL} -o ${MODEL_DIR}.tar.gz
cd ${TMP_DIR} && tar -xvf ${MODEL_NAME}.tar.gz

cd ${TF_MODELS_DIR}/research && \ 
${TF_MODELS_PY} object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=${CONFIG_FILE} \
--trained_checkpoint_prefix=${CHECKPOINT} \
--output_directory=${MODEL_DIR}/quant_exports \
--add_postprocessing_op=true

CI_DOCKER_EXTRA_PARAMS="-v ${MODEL_DIR}:/model" tensorflow/tools/ci_build/ci_build.sh CPU bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/model/tflite_graph.pb \
--output_file=/model/model_postprocessed_quantized_128_uint8.tflite \
--input_shapes=1,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops