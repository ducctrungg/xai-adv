# Paper

## Environment
- Python 3.10.5
- `pyenv`to manage environment

Snapshot project structure
```
├── amm.py
├── cicids2018_dl.py
├── cicids2018_ml.py
├── cnn.py
├── dataset
│   ├── test.csv
│   └── train.csv
├── defense.py
├── explainer.py
├── insdn.ipynb
├── log
│   ├── atk_ml_1.log
│   ├── atk_ml.log
│   ├── atk_mlp.log
│   ├── train_cnn.log
│   └── train_ml.log
├── main.py
├── mlp.py
├── model
│   ├── amm
│   │   ├── 0.2_patch.json
│   │   ├── 0.4_patch.json
│   │   ├── 0.6_patch.json
│   │   ├── 0.8_patch.json
│   │   ├── 1.0_patch.json
│   │   ├── dt_0.2_case0_patch.json
│   │   ├── dt_0.2_case1_patch.json
│   │   ├── dt_0.4_case0_patch.json
│   │   └── dt_0.4_case1_patch.json
│   ├── atk
│   │   ├── mlp_0.2
│   │   │   ├── assets
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       ├── variables.data-00000-of-00001
│   │   │       └── variables.index
│   │   ├── mlp_0.4
│   │   │   ├── assets
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       ├── variables.data-00000-of-00001
│   │   │       └── variables.index
│   │   ├── mlp_0.6
│   │   │   ├── assets
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       ├── variables.data-00000-of-00001
│   │   │       └── variables.index
│   │   ├── mlp_0.8
│   │   │   ├── assets
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       ├── variables.data-00000-of-00001
│   │   │       └── variables.index
│   │   └── mlp_1.0
│   │       ├── assets
│   │       ├── keras_metadata.pb
│   │       ├── saved_model.pb
│   │       └── variables
│   │           ├── variables.data-00000-of-00001
│   │           └── variables.index
│   └── ids
│       ├── cnn2
│       │   ├── assets
│       │   ├── keras_metadata.pb
│       │   ├── saved_model.pb
│       │   └── variables
│       │       ├── variables.data-00000-of-00001
│       │       └── variables.index
│       ├── cnn4
│       │   ├── assets
│       │   ├── keras_metadata.pb
│       │   ├── saved_model.pb
│       │   └── variables
│       │       ├── variables.data-00000-of-00001
│       │       └── variables.index
│       ├── lightgbm
│       └── rf
├── __pycache__
│   ├── amm.cpython-310.pyc
│   ├── cnn.cpython-310.pyc
│   ├── explainer.cpython-310.pyc
│   ├── mlp.cpython-310.pyc
│   └── utils.cpython-310.pyc
├── README.md
├── requirements.txt
├── utils.py
└── visualize.ipynb
```