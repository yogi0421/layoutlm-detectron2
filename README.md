# Convert LaoutLM Model to ONNX

## Requirements
```
onnx==1.8.0
protobuf==3.20.0
torch==1.10.1
opencv-python==4.5.5.64
onnxoptimizer
'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
layoutparser
```

## Update The package from `detectron2`
File location : 
```
.../python3.8/site-packages/detectron2/export/caffe2_export.py
```

replace :
```
import onnx.optimizer
```
to 
```
import onnxoptimizer as optimizer
```

## Run Model Convertion
```
python export_model.py \                                        
--sample-image /Users/yogiwahyu/Documents/9b1cfcdd-b1d6-4cb5-a1b7-91189995b9fc.jpg \
--config-file /Users/yogiwahyu/Documents/onnx-paddle-ocr/LayoutLM/config.yaml \
--output /Users/yogiwahyu/Documents/onnx-paddle-ocr/LayoutLM/output \
--export-method caffe2_tracing \
--format onnx \
MODEL.WEIGHTS /Users/yogiwahyu/Documents/onnx-paddle-ocr/LayoutLM/model_final.pth MODEL.DEVICE cpu
```

Logs : 
```
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/torchvision/io/image.py:11: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
WARNING:root:This caffe2 python run failed to load cuda module:No module named 'caffe2.python.caffe2_pybind11_state_gpu',and AMD hip module:No module named 'caffe2.python.caffe2_pybind11_state_hip'.Will run in CPU only mode.
[06/12 09:36:02 detectron2]: Command line arguments: Namespace(config_file='/Users/yogiwahyu/Documents/onnx-paddle-ocr/LayoutLM/config.yaml', export_method='caffe2_tracing', format='onnx', opts=['MODEL.WEIGHTS', '/Users/yogiwahyu/Documents/onnx-paddle-ocr/LayoutLM/model_final.pth', 'MODEL.DEVICE', 'cpu'], output='/Users/yogiwahyu/Documents/onnx-paddle-ocr/LayoutLM/output', run_eval=False, sample_image='/Users/yogiwahyu/Documents/9b1cfcdd-b1d6-4cb5-a1b7-91189995b9fc.jpg')
[06/12 09:36:02 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from /Users/yogiwahyu/Documents/onnx-paddle-ocr/LayoutLM/model_final.pth ...
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/torch/onnx/utils.py:267: UserWarning: `add_node_names' can be set to True only when 'operator_export_type' is `ONNX`. Since 'operator_export_type' is not set to 'ONNX', `add_node_names` argument will be ignored.
  warnings.warn("`{}' can be set to True only when 'operator_export_type' is "
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/torch/onnx/utils.py:267: UserWarning: `do_constant_folding' can be set to True only when 'operator_export_type' is `ONNX`. Since 'operator_export_type' is not set to 'ONNX', `do_constant_folding` argument will be ignored.
  warnings.warn("`{}' can be set to True only when 'operator_export_type' is "
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:32: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert tensor.dim() == 2 and tensor.size(-1) in [4, 5, 6], tensor.size()
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:399: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if num_classes + 1 == class_logits.shape[1]:
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:408: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert box_regression.shape[1] % box_dim == 0
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:409: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  cls_agnostic_bbox_reg = box_regression.shape[1] // box_dim == 1
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:427: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if input_tensor_mode:
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:457: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  nms_outputs = torch.ops._caffe2.BoxWithNMSLimit(
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:486: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
  for i, b in enumerate(int(x.item()) for x in roi_batch_splits_nms)
/Users/yogiwahyu/miniconda3/envs/py38/lib/python3.8/site-packages/detectron2/export/c10.py:486: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  for i, b in enumerate(int(x.item()) for x in roi_batch_splits_nms)
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::ResizeNearest type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::ResizeNearest type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::ResizeNearest type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::CollectRpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BatchPermutation type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BBoxTransform type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BBoxTransform type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::ResizeNearest type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::ResizeNearest type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::ResizeNearest type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::GenerateProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::CollectRpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::DistributeFpnProposals type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::RoIAlign type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BatchPermutation type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BBoxTransform type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BBoxTransform type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::BoxWithNMSLimit type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
WARNING: The shape inference of _caffe2::AliasWithName type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
[06/12 09:36:06 detectron2]: Success.
```