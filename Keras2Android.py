from keras.models import Sequential
from keras.models import model_from_json,model_from_yaml
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import os
import yolo3.model
# Load existing model.
with open("model.yaml",'r') as f:
    modelyaml = f.read()

model = model_from_yaml(modelyaml,custom_objects={'box_iou':yolo3.model.box_iou,'yolo_head':yolo3.model.yolo_head})
model.load_weights("model_weights.hdf5")

# All new operations will be in test mode from now on.
K.set_learning_phase(0)

# Serialize the model and get its weights, for quick re-building.
config = model.get_config()
weights = model.get_weights()

# Re-build a model where the learning phase is now hard-coded to 0.
#new_model = Sequential.from_config(config)
#new_model.set_weights(weights)

temp_dir = "graph"
checkpoint_prefix = os.path.join(temp_dir, "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"
output_optimized_graph_name= "opt_output_graph.pb"


# Temporary save graph to disk without weights included.
saver = tf.train.Saver()
checkpoint_path = saver.save(K.get_session(), checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
tf.train.write_graph(K.get_session().graph, temp_dir, input_graph_name)

input_graph_path = os.path.join(temp_dir, input_graph_name)
input_saver_def_path = ""
input_binary = False
#[node.op.name for node in model.outputs]
output_node_names = [node.op.name for node in model.outputs][0] # model dependent
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = os.path.join(temp_dir, output_graph_name)
clear_devices = False

# Embed weights inside the graph and save to disk.
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path,
                          output_node_names, restore_op_name,
                          filename_tensor_name, output_graph_path,
                          clear_devices, "")


input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_graph_path, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

#input_graph_def = tf.graph_util.remove_training_nodes(input_graph_def)


output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        [node.op.name for node in model.inputs], # an array of the input node(s)
        [output_node_names], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())



from tensorflow.python.tools import import_pb_to_tensorboard,

import_pb_to_tensorboard.import_to_tensorboard(output_graph_path,"logs/pb/")
import_pb_to_tensorboard.import_to_tensorboard(output_optimized_graph_name,"logs/pb_opt/")


#check summary

#bazel run tensorflow/tools/graph_transforms:summarize_graph -- --in_graph=/mnt/d/dev/keras-yolo3-jp2/opt_output_graph.pb

#bazel-bin/tensorflow/contrib/lite/toco/toco --input_file=/mnt/d/dev/keras-yolo3-jp2/graph/output_graph.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --output_file=/mnt/d/dev/keras-yolo3-jp2/graph/output.lite --inference_input_type=FLOAT --input_data_type=FLOAT --input_array=input_1_7 --output_arrays=separable_conv2d_10_5/BiasAdd --input_shape=1,224,224,3 --mean_value=127 --std_value=127
bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=/mnt/d/dev/keras-yolo3-jp2/graph/output_graph.pb \
--output_file=/mnt/d/dev/keras-yolo3-jp2/graph/output.lite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_type=QUANTIZED_UINT8 \
--input_array=input_1_7 \
--output_arrays=separable_conv2d_10_5/BiasAdd \
--input_shape=1,224,224,3 \
--mean_value=127.5 --std_value=127.5 \
--default_ranges_min=0 --default_ranges_max=6 \
--allow_nudging_weights_to_use_fast_gemm_kernel