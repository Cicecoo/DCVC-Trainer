# DCVC Trainer

## 用法

1. 拉取 https://github.com/microsoft/DCVC

2. 将 train.py、test_video.py 和 dvc_dataset.py 放入 DCVC/DCVC

3. 将 https://github.com/ZhihaoHu/PyTorchVideoCompression/ 的 DVC 文件夹放入 DCVC/DCVC

4. 修改 DCVC_net 的 forward，~~ 拿到 motioncompensation 的 prediction_init 作为 x_tilde ~~ 用参考帧（不提取特征）flow_warp 作为 x_tilde

