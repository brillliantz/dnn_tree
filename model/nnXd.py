# encoding: utf-8

import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd
from torch.nn.modules.utils import _single, _pair


class ConvXd(_ConvNd):
    """
    See doc of
        - torch.nn.modules.conv.Conv1d
        - torch.nn.modules.conv.Conv2d
        
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, dim=2):
        if dim == 1:
            repeat_func = _single
            self.conv_func = F.conv1d
        elif dim == 2:
            repeat_func = _pair
            self.conv_func = F.conv2d
        else:
            raise NotImplementedError("dim can only be 1 or 2. but received {:d}".format(dim))
        
        if padding == 'same' and stride == 1 and kernel_size % 2 == 0:
            padding = (kernel_size - 1) / 2
        
        kernel_size = repeat_func(kernel_size)
        stride      = repeat_func(stride)
        padding     = repeat_func(padding)
        dilation    = repeat_func(dilation)
        
        super(ConvXd, self).__init__(in_channels, out_channels,
                                     kernel_size, stride, padding, dilation,
                                     False, repeat_func(0), groups, bias)
    
    def forward(self, input_):
        return self.conv_func(input_, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)


class BatchNormXd(_BatchNorm):
    """
    See doc of
        - torch.nn.modules.batchnorm.BatchNorm1d
        - torch.nn.modules.batchnorm.BatchNorm2d
        
    """
    def _check_input_dim(self, input_):
        if input_.dim() != 2 and input_.dim() != 3 and input_.dim() != 4:
            raise ValueError('expected 2D or 3D or 4D input (got {}D input)'
                             .format(input_.dim()))


class MaxPoolXd(_MaxPoolNd):
    """
    See doc of parent class.
    
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, dim=2):
        super(MaxPoolXd, self).__init__(kernel_size,
                                        stride, padding, dilation,
                                        return_indices, ceil_mode)
        if dim == 1:
            self.pool_func = F.max_pool1d
        elif dim == 2:
            self.pool_func = F.max_pool2d
        else:
            raise NotImplementedError("dim can only be 1 or 2. but received {:d}".format(dim))
    
    def forward(self, input_):
        return self.pool_func(input_, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
    
    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class AvgPoolXd(_AvgPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, dim=2):
        super(AvgPoolXd, self).__init__()
        
        if dim == 1:
            self.kernel_size = _single(kernel_size)
            self.stride = _single(stride if stride is not None else kernel_size)
            self.padding = _single(padding)
            self.pool_func = F.avg_pool1d
        elif dim == 2:
            self.kernel_size = kernel_size
            self.stride = stride or kernel_size
            self.padding = padding
            self.pool_func = F.avg_pool2d
        else:
            raise NotImplementedError("dim can only be 1 or 2. but received {:d}".format(dim))
        
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
    
    def forward(self, input_):
        return self.pool_func(input_, self.kernel_size, self.stride,
                              self.padding, self.ceil_mode, self.count_include_pad)
