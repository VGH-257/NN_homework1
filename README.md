# NN_homework1
本项目基于 numpy 实现了三层神经网络分类器，对数据集 CIFAR-10 进行训练和测试。  

## 文件结构
```bash
NN_homework1/
│
├── data/ # path of CIFAR dataset (automatically download using torchvision)
├── ckpt/ # path of checkpoint of models
├── configs/ # configs for simulation and evaluation
└── code/ 
    ├── new_log/ # path of training logs
    ├── data_loader.py # dataloader
    ├── mlp.py # model
    ├── train_mlp.py # main file
    ├── analyze.py # draw figures with different hyperparameters
    ├── vis.py # visualization of parameters
    └── utils.py # tools
```

## 训练测试说明
可调用train_mlp.py 进行训练(train函数)和测试(test函数)，示例command见code/run_train.sh，参数网格搜索见code/grid_search.sh。

## 模型权重
模型权重可至[link](https://pan.baidu.com/s/12YZk9g53JGJWCJ-rF33YXg?pwd=ad7q)下载。（提取码: ad7q）
