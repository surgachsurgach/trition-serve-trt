### a. Using the NGC PyTorch container

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

### b. Run compiling model inside the container
```
$ cd /Torch-TensorRT/
$ pip install -r data/ml/workspace/triton-serve-trt/requirements.txt
$ PYTHONPATH=. python data/ml/workspace/triton-serve-trt/comile_trt.py
```

And navigate a web browser to the IP address or hostname of the host machine
at port 8888: ```http://[host machine]:8888```

Use the token listed in the output from running the jupyter command to log
in, for example:

```http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b```

### c. Run triton server
We use the pytorch backend (pytorch_libtorch) for deploying the model. To install Triton, follow the instructions [here](https://github.com/triton-inference-server/pytorch_backend)
```bash
$ cd data/ml/workspace/triton-serve-trt/
$ docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD:/models nvcr.io/nvidia/tritonserver:24.05-py3 
$ tritonserver --model-repository=/models
```

### d. Run triton client
```bash
$ docker run --gpus=all --rm -it -v $PWD:/Torch-TensorRT --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:24.05-py3 bash
$ pip install tritonclient[all]
$ cd /Torch-TensorRT
$ python data/ml/workspace/triton-serve-trt/triton_client.py
```