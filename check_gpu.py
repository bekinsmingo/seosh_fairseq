import torch
#import apex

print('cuda availability ? {}'.format(torch.cuda.is_available()))
print('total gpu nums : {}'.format(torch.cuda.device_count()))
print('cudnn backends version : {}'.format(torch.backends.cudnn.version()))
print('cuda version : {}'.format(torch.version.cuda))

print('*'*30)

for n in range(torch.cuda.device_count()):
    print('{}th GPU name is {}'.format(n,torch.cuda.get_device_name(n)))
    print('\t capability of this GPU is {}'.format(torch.cuda.get_device_capability(n)))
