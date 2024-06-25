## Using the NGC PyTorch container

At this point, we recommend pulling the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
from [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) as follows:

```
docker pull nvcr.io/nvidia/pytorch:24.05-py3
```

Replace ```22.05``` with a different string in the form ```yy.mm```,
where ```yy``` indicates the last two numbers of a calendar year, and
```mm``` indicates the month in two-digit numerical form, if you wish
to pull a different version of the container.

The NGC PyTorch container ships with the Torch-TensorRT tutorial notebooks.
Therefore, you can run the container and the notebooks therein without
mounting the repo to the container. To do so, run

```
cd $PROJECT_ROOT
docker run --gpus=all --rm -it -v $PWD:/Torch-TensorRT \
--net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
nvcr.io/nvidia/pytorch:24.05-py3 bash
```

### a. Run compiling model inside the container
```
$ cd /Torch-TensorRT/
$ pip install -r data/ml/workspace/triton-serve-trt/requirements.txt
$ PYTHONPATH=. python data/ml/workspace/triton-serve-trt/comile_trt.py
```

### b. Run triton server
We use the pytorch backend (pytorch_libtorch) for deploying the model. To install Triton, follow the instructions [here](https://github.com/triton-inference-server/pytorch_backend)
```bash
$ cd data/ml/workspace/triton-serve-trt/
$ docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD:/models nvcr.io/nvidia/tritonserver:24.05-py3 
$ tritonserver --model-repository=/models
```

### c. Run triton client
```bash
$ docker run --gpus=all --rm -it -v $PWD:/Torch-TensorRT --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:24.05-py3 bash
$ pip install tritonclient[all]
$ cd /Torch-TensorRT
$ python data/ml/workspace/triton-serve-trt/triton_client.py
```

## References
- https://github.com/pytorch/TensorRT/tree/main
- https://pytorch.org/TensorRT/user_guide/saving_models.html
- https://pytorch.org/TensorRT/tutorials/serving_torch_tensorrt_with_triton.html
- https://github.com/pytorch/TensorRT/tree/main/notebooks
- https://github.com/pytorch/TensorRT/blob/v2.3.0-rc1/py/torch_tensorrt/_compile.py
- https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups
- https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_1-model_deployment/client.py
- https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_infer_client.py
- 
## Version Compatibility
- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags
- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
- https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-24-02.html#rel-24-02
- https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html
- https://docs.nvidia.com/deploy/cuda-compatibility/#forward-compatibility