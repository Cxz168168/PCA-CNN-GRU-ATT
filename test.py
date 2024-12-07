import torch

# 检查 GPU 是否可用
if torch.cuda.is_available():
    print("GPU 可用，当前使用的设备为：", torch.cuda.get_device_name(0))
else:
    print("GPU 不可用，当前使用的是 CPU。")
