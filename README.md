* 训练 DDPM
    ```
    python ddpm_mnist.py --mode train --epochs 10
    ```
* 训练噪声域分类器（3~5 个 epoch 就够 MNIST）
    ```
    python ddpm_mnist.py --mode train_clf --epochs 3
    ```
* 普通采样（36 张）
    ```
    python ddpm_mnist.py --mode sample --n 36 --save ./ddpm_out/test.png
    ```
* 分类器引导采样：全部生成“3”
    ```
    python ddpm_mnist.py --mode sample_guided --digit 3 --scale 20.0 --n 36 --save ./ddpm_out/guided_3.png
    ```