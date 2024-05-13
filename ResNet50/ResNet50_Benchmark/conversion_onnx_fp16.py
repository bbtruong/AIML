import onnx
from onnxconverter_common import float16

model = onnx.load("resnet50_model.onnx")
model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False, disable_shape_infer=False, op_block_list=None, node_block_list=None)
onnx.save(model_fp16, "resnet50_model_fp16.onnx")