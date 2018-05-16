import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .nnXd import ConvXd, MaxPoolXd, AvgPoolXd, BatchNormXd


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dim=2):
    """
    3x3 convolution with padding
    
    Parameters
    ----------
    in_planes : int
    out_planes : int
    stride : int
    dim : {1, 2}

    Returns
    -------
    ConvXd
    
    Notes
    -----
    if stride == 1:
        (in_len - kernel_size + 2 * padding) / stride + 1 =
        (in_len - 3 + 2 * 1) / 1 + 1 =
        in_len

    """
    return ConvXd(in_planes, out_planes,
                  kernel_size=3, stride=stride, padding=1,
                  bias=False, dim=dim)


class _BaseBlock(nn.Module):
    """
    Abstract Base Class (ABC)
    
    Parameters
    ----------
    in_planes : int
        (in_channels)
    
    
    """
    def __init__(self, in_planes, out_planes,
                 stride=1,
                 dim=2):
        super(_BaseBlock, self).__init__()
        
        self._create_operations(in_planes, out_planes, stride, dim)

    def _create_operations(self, in_planes, out_planes, stride, dim):
        raise NotImplementedError()
    
    def forward(self, x):
        pass


def get_shortcut_operation(in_planes, out_planes, stride, expansion, dim):
    if stride != 1 or in_planes != out_planes * expansion:
        res = nn.Sequential(
                ConvXd(in_planes, out_planes * expansion,
                       kernel_size=1, stride=stride, bias=False,
                       dim=dim),
                nn.BatchNorm2d(out_planes * expansion),
        )
    else:
        def res(x):
            return x
    return res


class BasicBlock(_BaseBlock):
    """
    -- The basic residual layer block for 18 and 34 layer network, and the
    -- The bottleneck residual layer for 50, 101, and 152 layer networks
    
    """
    expansion = 1
    
    def _create_operations(self, in_planes, out_planes, stride, dim):
        """
        If stride = 1, then out_len = in_len;
        If stride = 2, then out_len = in_len / 2;
        
        Parameters
        ----------
        in_planes : int
            (in_channels)
        out_planes
        stride
        dim

        """
        # only this conv can halve the in_len
        self.conv1 = conv3x3(in_planes, out_planes, stride, dim=dim)
        self.bn1 = BatchNormXd(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(out_planes, out_planes, stride=1,
                             dim=dim)
        self.bn2 = BatchNormXd(out_planes)
        
        self.shortcut = get_shortcut_operation(in_planes, out_planes,
                                               stride, self.expansion,
                                               dim)
        
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class PreActBasicBlock(_BaseBlock):
    """
    -- The basic residual layer block for 18 and 34 layer network, and the
    -- The bottleneck residual layer for 50, 101, and 152 layer networks
    
    """
    expansion = 1
    
    def _create_operations(self, in_planes, out_planes, stride, dim):
        """
        If stride = 1, then out_len = in_len;
        If stride = 2, then out_len = in_len / 2;
        
        Parameters
        ----------
        in_planes : int
            (in_channels)
        out_planes
        stride
        dim

        """
        # only this conv can halve the in_len
        self.bn1 = BatchNormXd(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride, dim=dim)

        self.bn2 = BatchNormXd(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride=1,
                             dim=dim)
        
        self.shortcut = get_shortcut_operation(in_planes, out_planes,
                                               stride, self.expansion,
                                               dim)
        
        self.stride = stride
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += self.shortcut(x)
        
        return out


class BottleneckBlock(_BaseBlock):
    """
    -- The basic residual layer block for 18 and 34 layer network, and the
    -- The bottleneck residual layer for 50, 101, and 152 layer networks
    
    """
    expansion = 4
    
    def _create_operations(self, in_planes, out_planes, stride, dim):
        """
        
        Parameters
        ----------
        in_planes : int
            (in_channels)
        out_planes
        stride
        dim

        """
        self.conv1 = ConvXd(in_planes, out_planes,
                            kernel_size=1, stride=1, padding=0,
                            bias=False,
                            dim=dim)
        self.bn1 = BatchNormXd(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # only this conv can halve the in_len
        self.conv2 = ConvXd(out_planes, out_planes,
                            kernel_size=3, stride=stride, padding=1,
                            bias=False,
                            dim=dim)
        self.bn2 = BatchNormXd(out_planes)

        out_planes_3 = out_planes * 4
        self.conv3 = ConvXd(out_planes, out_planes_3,
                            kernel_size=1, stride=1, padding=0,
                            bias=False,
                            dim=dim)
        self.bn3 = BatchNormXd(out_planes_3)

        self.shortcut = get_shortcut_operation(in_planes, out_planes,
                                               stride, self.expansion,
                                               dim)
        
        self.stride = stride
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(x)
        
        out = self.relu(out)
        
        return out


class PreActBottleneckBlock(_BaseBlock):
    """
    -- The basic residual layer block for 18 and 34 layer network, and the
    -- The bottleneck residual layer for 50, 101, and 152 layer networks
    
    """
    expansion = 4
    
    def _create_operations(self, in_planes, out_planes, stride, dim):
        """
        
        Parameters
        ----------
        in_planes : int
            (in_channels)
        out_planes
        stride
        dim

        """
        self.bn1 = BatchNormXd(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvXd(in_planes, out_planes,
                            kernel_size=1, stride=1, padding=0,
                            bias=False,
                            dim=dim)
        
        # only this conv can halve the in_len
        self.bn2 = BatchNormXd(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvXd(out_planes, out_planes,
                            kernel_size=3, stride=stride, padding=1,
                            bias=False,
                            dim=dim)
        
        self.bn3 = BatchNormXd(out_planes)
        self.relu = nn.ReLU(inplace=True)
        out_planes_3 = out_planes * 4
        self.conv3 = ConvXd(out_planes, out_planes_3,
                            kernel_size=1, stride=1, padding=0,
                            bias=False,
                            dim=dim)
        
        self.shortcut = get_shortcut_operation(in_planes, out_planes,
                                               stride, self.expansion,
                                               dim)
        
        self.stride = stride
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        out += self.shortcut(x)
        
        return out


class ResNet(nn.Module):
    """
    (i)  for the same output feature map size, the layers have the same number of filters;
    (ii) if the feature map size is halved, the number of filters is doubled
         so as to preserve the time complexity per layer.
    We perform downsampling directly by convolutional layers that have a stride of 2
    
    """

    def __init__(self, in_planes, num_classes, block_cls, num_blocks, dim):
        super(ResNet, self).__init__()

        self.in_planes = in_planes
        exp_base = 2
        
        mode = 'imagenet'
        if mode == 'imagenet':
            base_planes = 64
            num_planes = [base_planes * exp_base**i for i in range(4)]
            self.conv0 = ConvXd(self.in_planes, num_planes[0],
                                kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn0 = BatchNormXd(num_planes[0])
            self.relu0 = nn.ReLU(inplace=True)  # size/2
            self.maxpool0 = MaxPoolXd(kernel_size=3, stride=2, padding=1, dim=dim)  # size/2
            
            self.top_layer = nn.Sequential(self.conv0, self.bn0, self.relu0, self.maxpool0)
        elif mode == 'cifar':
            # imagenet mode is complete, while cifar mode is not the same as publication
            base_planes = 16
            num_planes = [base_planes * math.pow(exp_base, i) for i in range(4)]
            self.conv0 = ConvXd(self.in_planes, num_planes[0],
                                kernel_size=3, stride=1, padding=1,
                                bias=False)
            self.bn0 = BatchNormXd(num_planes[0])
            self.relu0 = nn.ReLU(inplace=True)  # size/2
            
            self.top_layer = nn.Sequential(self.conv0, self.bn0, self.relu0)
        else:
            raise NotImplementedError("mode = {:s}".format(mode))

        self._ip = num_planes[0]
        
        self.layer1 = self._make_layer(block_cls, num_planes[0], num_blocks[0])
        self.layer2 = self._make_layer(block_cls, num_planes[1], num_blocks[1], stride=exp_base)  # size/2
        self.layer3 = self._make_layer(block_cls, num_planes[2], num_blocks[2], stride=exp_base)  # size/2
        self.layer4 = self._make_layer(block_cls, num_planes[3], num_blocks[3], stride=exp_base)  # size/2
        
        # TODO kernel_size need to be adaptable
        self.avgpool = AvgPoolXd(7, stride=1, dim=dim)
        self.fc = nn.Linear(num_planes[3] * block_cls.expansion, num_classes)

        self._init_weights()
        
    def _init_weights(self):
        """
        Fills the input `Tensor` with values according to the method
        described in "Delving deep into rectifiers: Surpassing human-level
        performance on ImageNet classification" - He, K. et al. (2015)
        
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                width = m.kernel_size[0]
                height = m.kernel_size[1]
                n = width * height * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block_cls, out_planes, n_blocks, stride=1):
        if n_blocks < 1:
            return nn.Sequential()
        
        layers = list()
        layers.append(block_cls(self._ip, out_planes, stride))
        self._ip = out_planes * block_cls.expansion
        for i in range(1, n_blocks):
            layers.append(block_cls(self._ip, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.top_layer(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet18(dim, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        dim : {1, 2}
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(dim=dim, block_cls=BasicBlock, num_blocks=[2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
