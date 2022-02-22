# Requirements and Installation

## 1. Pytorch, Python

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

* check pytorch version

```bash
root@b96a5cf449f7:/workspace# python -c "import torch; print(torch.__version__); print(torch.cuda.is_available());"
1.9.0+cu111
True
```

* in detail

```python
import torch
import apex

print('cuda availability ? {}'.format(torch.cuda.is_available()))
print('total gpu nums : {}'.format(torch.cuda.device_count()))
print('cudnn backends version : {}'.format(torch.backends.cudnn.version()))
print('cuda version : {}'.format(torch.version.cuda))

print('*'*30)

for n in range(torch.cuda.device_count()):
 print('{}th GPU name is {}'.format(n,torch.cuda.get_device_name(n)))
 print('\t capability of this GPU is {}'.format(torch.cuda.get_device_capability(n)))
```

* in my case

```python
cuda availability ? True
total gpu nums : 8
cudnn backends version : 8005
cuda version : 11.1
******************************
0th GPU name is Tesla P40
         capability of this GPU is (6, 1)
1th GPU name is Tesla P40
         capability of this GPU is (6, 1)
2th GPU name is Tesla P40
         capability of this GPU is (6, 1)
3th GPU name is Tesla P40
         capability of this GPU is (6, 1)
4th GPU name is Tesla P40
         capability of this GPU is (6, 1)
5th GPU name is Tesla P40
         capability of this GPU is (6, 1)
6th GPU name is Tesla P40
         capability of this GPU is (6, 1)
7th GPU name is Tesla P40
         capability of this GPU is (6, 1)
```



## 2. Fairseq (To install fairseq** and develop locally)

``` bash
git clone https://github.com/pytorch/seosh_fairseq
cd seosh_fairseq
pip install --editable ./
```

* check

```bash
python3 -c "import fairseq; from fairseq.examples.speech_recognition.w2l_decoder import W2lDecoder;"
```

* if u got error like this...

```python
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/workspace/fairseq/examples/speech_recognition/__init__.py", line 1, in <module>
    from . import criterions, models, tasks  # noqa
  File "/workspace/fairseq/examples/speech_recognition/criterions/__init__.py", line 15, in <module>
    importlib.import_module(
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ModuleNotFoundError: No module named 'examples'
```

```bash
cd $FAIRSEQ_ROOT &&\
python setup.py build develop
```

* example run using fairseq

```bash
cd $FAIRSEQ_ROOT/tests/speech &&\
python3 test_wav2vec2.py
```

## 3. Apex (cuda major capability<7 (e.g. p40) 인 경우 fp16이 효과가 없을 수 있음)

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## 4. additional

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .
 
```bash
pip install PyArrow
```

* **huggingface**

you need to use transformers

```
pip install transformers
pip install datasets
```



# Official Fairseq Doc

* [full documentation](https://fairseq.readthedocs.io/) 

# Pre-trained models and examples

* [Translation](examples/translation/README.md): convolutional and transformer models are available
* [Language Modeling](examples/language_model/README.md): convolutional and transformer models are available


```
docker run --gpus all -it --name tmp \
-v /home1/irteam/users/seosh/decoder_pratice:/workspace \
-v /mnt/clova_speech:/mnt/clova_speech \
--shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
--device=/dev/snd flml/flashlight:cuda-latest
```
