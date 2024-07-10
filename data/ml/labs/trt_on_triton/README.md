## Hosting Torch-Tensorrt on Triton Server using the NGC container

We recommend pulling the images from [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) as follows:

```
docker pull nvcr.io/nvidia/pytorch:24.05-py3
docker pull nvcr.io/nvidia/tritonserver:24.05-py3 
```

Replace ```24.05``` with a different string in the form ```yy.mm```,
where ```yy``` indicates the last two numbers of a calendar year, and
```mm``` indicates the month in two-digit numerical form, if you wish
to pull a different version of the container.

*FYI. The NGC PyTorch container ships with the Torch-TensorRT tutorial notebooks.

### 1. Run compiling model
```bash
$ cd $PROJECT_ROOT (the root directory of repo)
$ docker run --gpus=all --rm -it -v $PWD:/workspace \
    --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:24.05-py3
$ cd /workspace
$ export MODEL_NAME=bert4rec
$ pip install -r data/ml/labs/trt_on_triton/requirements.additional.txt
$ PYTHONPATH=. python data/ml/labs/trt_on_triton/${MODEL_NAME}/compiler.py
```

### 2. Run triton server hosting compiled model
*FYI. We use the pytorch backend (pytorch_libtorch) for model backend
```bash
$ cd data/ml/labs/trt_on_triton/${MODEL_NAME}
$ docker run --gpus=all --rm -it -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD:/workspace nvcr.io/nvidia/tritonserver:24.05-py3
$ cd /workspace
$ tritonserver --model-repository=./model_repository
```

### 3. Run triton inference client
```bash
$ cd data/ml/labs/trt_on_triton/${MODEL_NAME}
$ docker run --gpus=all --rm -it -v $PWD:/workspace --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:24.05-py3
$ cd /workspace
$ pip install tritonclient[all]
$ python client.py
```

### 4. Run Performance Analyzer
```bash
$ cd data/ml/labs/trt_on_triton/${MODEL_NAME}
$ docker run --gpus all --rm -it --net host -v $PWD:/workspace nvcr.io/nvidia/tritonserver:24.05-py3
$ cd /workspace
$ pip install tritonclient[perf_analyzer]
$ perf_analyzer -m bert4rec -u 127.0.0.1:8001 -i grpc --input-data ./test_data/data.json --request-rate-range 1500:3000:500 -f perf_analyzer_result.csv
```

## References
- https://github.com/pytorch/TensorRT/tree/main
- https://github.com/pytorch/TensorRT/blob/main/py/README.md
- https://pytorch.org/TensorRT/user_guide/saving_models.html
- https://pytorch.org/TensorRT/tutorials/serving_torch_tensorrt_with_triton.html
- https://github.com/pytorch/TensorRT/tree/main/notebooks
- https://github.com/pytorch/TensorRT/blob/v2.3.0-rc1/py/torch_tensorrt/_compile.py
- https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups
- https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_1-model_deployment/client.py
- https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_infer_client.py
- https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/quick_start.md

## Version Compatibility
- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags
- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
- https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-24-05.html#rel-24-05
- https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.html
- https://docs.nvidia.com/deploy/cuda-compatibility/#forward-compatibility