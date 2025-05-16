# DCVC Trainer

## 结果

目前只在 quality=1 (lambda=256) 配置下得到与[预训练权重](https://github.com/microsoft/DCVC?tab=readme-ov-file#clipboard-dcvc-family)接近的结果。

|P frame 指标|DCVC-TCM single 策略<br>+余弦学习率|dcvc_quality_0_psnr<br> (λ = 256)|
| --- | --- | --- |
|bpp|0.02645|0.03060|
|bpp_mv_y|0.00629|0.00830|
|bpp_mv_z|0.00013|0.00045|
|bpp_y|0.01964|0.02082|
|bpp_z|0.00038|0.00102|
|psnr|32.0844|32.9327|


## 说明

### 模型代码

对应 src 文件夹，来自 [DCVC24年仓库](https://github.com/microsoft/DCVC/tree/4df94295c8dbe0a26456582d1a0eddb3465f1597/DCVC)。

其中需要添加对量化操作的训练时处理，如
```py
    def quant(self, x, force_detach=True):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n
        return torch.round(x)
```
或
```py
    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        noise = noise.clone().detach()
        return x + noise
```

### 数据集

使用 [DVC](https://github.com/ZhihaoHu/PyTorchVideoCompression/blob/master/DVC/dataset.py) 的数据集配置。
test.txt 来自[此处](https://github.com/ZhihaoHu/PyTorchVideoCompression/blob/master/DVC/data/vimeo_septuplet/test.txt)。

### 光流网络

Spynet 预训练权重的加载同样参考 [DVC](https://github.com/ZhihaoHu/PyTorchVideoCompression/blob/master/DVC/subnet/endecoder.py#L313)，权重来自[此处](https://github.com/ZhihaoHu/PyTorchVideoCompression/tree/master/DVC/examples/flow_pretrain_np)。


