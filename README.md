

## Experiments



> 📋 Please download and place all datasets into the data directory. For Clohting1M, please run "python ClothingData.npy" to generate a data file.

To train RLM without semi on CIFAR-10/100

```
python Train_cifar.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5 --step 200 --n 6
```

```
python Train_cifar.py --dataset cifar100 --noise_type instance --noise_rate 0.4 --step 20 --n 6
```

To train RLM with semi on CIFAR-10/100

```
python Train_cifar_semi.py --dataset cifar10 --noise_type instance --noise_rate 0.4  --lambda_u 15 --step 200 --n 6
```

```
python Train_cifar_semi.py --dataset cifar100 --noise_type pairflip --noise_rate 0.45  --lambda_u 15 --step 30 --n 6
```

To train RLM on Clothing1M

```train Clothing1M
python Train_Clothing1M.py
```
To train RLM on Webvision

```train Clothing1M
python Train_webvision.py
```




