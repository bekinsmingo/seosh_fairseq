# Requirements and Installation

## 1. Pytorch, Python

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)



## 2. Fairseq (To install fairseq and develop locally)

``` bash
git clone https://github.com/SeunghyunSEO/seosh_fairseq
cd seosh_fairseq
pip install --editable ./
```

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
total gpu nums : 4
cudnn backends version : 8200
cuda version : 11.3
******************************
0th GPU name is NVIDIA A100-SXM4-80GB
         capability of this GPU is (8, 0)
1th GPU name is NVIDIA A100-SXM4-80GB
         capability of this GPU is (8, 0)
2th GPU name is NVIDIA A100-SXM4-80GB
         capability of this GPU is (8, 0)
3th GPU name is NVIDIA A100-SXM4-80GB
         capability of this GPU is (8, 0)
```

* check fairseq installation

```
root@ae753a5ee1a0:/usr/local/lib/python3.8/dist-packages# cat /usr/local/lib/python3.8/dist-packages/fairseq.egg-link 
/workspace/seosh_fairseq
```

```bash
root@ae753a5ee1a0:/workspace/seosh_fairseq# python3 -c "import fairseq; from fairseq.models.transformer_lm import TransformerLanguageModel; print(TransformerLanguageModel)"
2022-02-22 09:54:16 | INFO | fairseq.tasks.text_to_speech | Please install tensorboardX: pip install tensorboardX
<class 'fairseq.models.transformer_lm.TransformerLanguageModel'>
root@ae753a5ee1a0:/workspace/seosh_fairseq# 
```

* example run using fairseq

```
root@ae753a5ee1a0:/workspace/fairseq/tests# python3 test_roberta.py
2022-02-22 09:17:33 | INFO | fairseq.tasks.text_to_speech | Please install tensorboardX: pip install tensorboardX
root@ae753a5ee1a0:/workspace/fairseq/tests# 
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

