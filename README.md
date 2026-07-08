# ComfyUI DCT Compression Nodes

This custom node package adds a small PyTorch model pipeline for loading,
compressing, saving, and running CNNs with a frequency-domain compression
scheme inspired by *Compressing Convolutional Neural Networks*.

The compressed model remains usable for inference because each compressed layer
reconstructs its weight tensor during `forward()`.

## Nodes

### Load Torch CNN Model
Loads a pickled `torch.nn.Module` from a `.pt` or `.pth` file.

Inputs:
- `model_path`: path to a serialized PyTorch module object
- `map_location`: device string such as `cpu` or `cuda:0`

### Load Torch Model From State Dict
Loads a checkpoint or raw `state_dict` using a model class path.

Inputs:
- `checkpoint_path`: path to the checkpoint file
- `model_class_path`: import path or file path to a model constructor
- `model_args_json`: JSON list of positional constructor arguments
- `model_kwargs_json`: JSON object of keyword constructor arguments
- `state_dict_key`: optional nested key such as `state_dict` or `model_state_dict`
- `map_location`: device string such as `cpu` or `cuda:0`
- `strict`: whether to require an exact key match when loading weights

### Load TorchVision ResNet18
Loads a torchvision `resnet18` model, optionally with ImageNet pretrained
weights.

Inputs:
- `pretrained`: load `ResNet18_Weights.IMAGENET1K_V1` when enabled

If `pretrained` is enabled, torchvision will download the weights the first
time the node runs unless they are already cached locally.

### DCT Compress CNN
Replaces `nn.Conv2d` and `nn.Linear` layers with DCT-compressed wrappers.
Lower-frequency bands receive more of the parameter budget than higher-frequency
bands.

Inputs:
- `model`: the loaded PyTorch model
- `compression_ratio`: target ratio for shared DCT parameters
- `alpha`: frequency-band weighting shape parameter
- `beta`: frequency-band weighting shape parameter
- `seed`: deterministic hash seed
- `include_conv`: compress convolution layers
- `include_linear`: compress linear layers
- `freeze_parameters`: freeze the shared DCT parameters after compression

### Save Torch Model
Saves the current model object, including DCT-compressed wrappers, to disk.

Inputs:
- `model`: the PyTorch model to save
- `output_path`: directory or full file path
- `file_name`: output file name when `output_path` is a directory
- `save_to_cpu`: move the model to CPU before saving

### Run Torch Model Inference
Runs a `PYTORCH_MODEL` on an input tensor and returns the output tensor.

Inputs:
- `model`: the PyTorch model to execute
- `input_tensor`: input tensor for the model
- `return_to_cpu`: return the output on CPU
- `use_no_grad`: run inference under `torch.no_grad()`

## Usage

1. Load a model with `Load Torch CNN Model`, or load a checkpoint with
   `Load Torch Model From State Dict`, or load a pretrained backbone with
   `Load TorchVision ResNet18`.
2. Connect the model to `DCT Compress CNN`.
3. Optionally save the compressed model with `Save Torch Model`.
4. Use `Run Torch Model Inference` to execute the model on a tensor.

## Example: Loading a State Dict

If your model class is `my_models.SmallCNN`, configure the loader like this:

```text
model_class_path = my_models.SmallCNN
model_args_json = []
model_kwargs_json = {"num_classes": 10}
state_dict_key = state_dict
```

If the checkpoint stores weights under a different key, set `state_dict_key`
accordingly, for example `model_state_dict`.

## Notes

- You can load either a serialized `torch.nn.Module` or a checkpoint containing
  a `state_dict`.
- The compression is frequency-sensitive and favors lower-frequency bands.
- Set `freeze_parameters=False` if you want the shared DCT parameters to stay
  trainable for fine-tuning.
- The inference node expects the model output to be a single `torch.Tensor`.
