# Computer Vision 2025 Homework4

## Setting Up

```sh
# Clone the repository
git clone https://github.com/liuhanzuo/pytorch-segmentation.git
cd pytorch-segmentation
conda create -n cvhw4 python=3.8
conda activate cvhw4
pip install -r requirements.txt
```

Note that my test environment is 12.4 for nvidia driver version, it is not verified for other nvidia version's correntness

## Training
```json
{
    "name": "SCTNet-resnet152-CityScapes", run name, saved in ./saved
    "n_gpu": 1, gpu number
    "use_synch_bn": false,

    "arch": {
        "type": "SCTNet",
        "args": {
            "backbone": "resnet152", backbone name, see more in models/resnet.py
            "freeze_bn": false,
            "freeze_backbone": false
        }
    },

    "train_loader": {
        "type": "CityScapes",
        "args":{
            "data_dir": "/root/autodl-tmp/data/cityscapes", dataset dir
            "batch_size": 32, training batchsize, 32 is for 96G GPU memory, fit your training sitiuation
            "base_size": 400,
            "crop_size": 380,
            "augment": true,
            "shuffle": true,
            "scale": true,
            "flip": true,
            "rotate": true,
            "blur": false,
            "split": "train",
            "num_workers": 8
        }
    },

    "val_loader": {
        "type": "CityScapes",
        "args":{
            "data_dir": "/root/autodl-tmp/data/cityscapes",
            "batch_size": 32, validation batchsize
            "crop_size": 480,
            "val": true,
            "split": "val",
            "num_workers": 4
        }
    },

    "optimizer": {
        "type": "SGD",
        "differential_lr": true,
        "args":{
            "lr": 1e-2, 
            "weight_decay": 5e-6,
            "momentum": 0.9
        }
    },

    "loss": "HybridLoss",
    "ignore_index": 255,
    "lr_scheduler": { lr scheduler
        "type": "CosineWithMinLR",
        "args": {
            "warmup_epochs": 10,
            "min_lr": 5e-5
        }
    },

    "trainer": {
        "epochs": 100, total training epoch
        "save_dir": "saved/",
        "save_period": 10, saving interval
  
        "monitor": "max Mean_IoU",
        "early_stop": 10,
        
        "tensorboard": true,
        "log_dir": "saved/runs",
        "log_per_iter": 20,

        "val": true,
        "val_per_epochs": 5 validation interval
    }
}

```

The notes are placed on the above json format file, you could modify it by yourself. It takes about 4 hour for my config

## Evaluation
load your model path and config path in the saved/ directory and run
```python
python evaluation.py --model /path/to/your/model.pth --config /path/to/your/config.json
```

Choose best_model.pth for evaluating the best model during training. it will output pixel acc and mIOU

## Inference
Use the following instruction to show the inference result.
```python
python inference.py --model /path/to/your/model.pth --config /path/to/your/config.json
```

Have fun with it!

## Example
here is an inference example
![Inference Example](eval_results/val_samples.png)