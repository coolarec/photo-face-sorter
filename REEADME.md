# 运行方法

放到 ``/model/dataset`` 里面
目录结构示例如下
```commandline
/CODE/FACE
├─model
│  ├─dataset
│  │  ├─000
│  │  ├─001
│  │  ├─002
│  │  ├─003
│  │  ├─004
│  │  ├─005
│  │  ├─006
│  │  └─007
│  └─__pycache__
├─result
├─static
│  └─uploads
└─templates
```

运行 ``train.py`` 自动生成模型并保存。
直接运行 ``main.py`` 可以自动调用生成好的模型来识别人脸