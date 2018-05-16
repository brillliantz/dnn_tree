- mean, std: batch-wise v.s. rolling
- use pre-activation ResNet instead of original ResNet
- Is BatchNorm appropriate?
- Kaiming Init:
  Fills the input `Tensor` with values according to the method
  described in "Delving deep into rectifiers: Surpassing human-level
  performance on ImageNet classification" - He, K. et al. (2015)
  ```python
  # way 1
  for m in self.modules():
      if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)

  # way 2
  for m in self.modules():
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
  ```