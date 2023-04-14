## Usage

### 1. generate input data

```
python generate_inputs.py
```

### 2. generate output of paddle develop

```
pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.develop.whl
python run.py --tag=paddle_dev
pip uninstall paddlepaddle_gpu
```

### 3. generate output of paddle release

```
pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.incubate.whl
python run.py --tag=paddle_rel
pip uninstall paddlepaddle_gpu
```

### 4. generate output of pytorch

```
pip install torch
python run.py --tag=torch
pip uninstall torch
```

### 5. check the results

```
python check_results.py
```
