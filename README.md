# clothing1M_experiments
some experiments on learning with noisy data using the clothing1M datset.     
Paper --> [Label-Noise Robust Generative Adversarial Networks](https://arxiv.org/abs/1811.11165)


## Download Clothing1M Dataset
To download the Clothing1M dataset, checkout this repo -- https://github.com/Cysu/noisy_label#clothing1m-experiments

Once downloaded ensure the data is arranged as shown below --
```
/dataset_dir
├── category_names_chn.txt
├── category_names_eng.txt
├── clean_label_kv.txt
├── clean_test_key_list.txt
├── clean_train_key_list.txt
├── clean_val_key_list.txt
├── images
│   ├── 0
│   ├── ⋮
│   └── 9
├── noisy_label_kv.txt
├── noisy_train_key_list.txt
├── README.md
└── venn.png
```

## Data Loader
The dataloader is written as a pytorch lightning data module.      

If pytorch lightning is not used the data loader can be loaded as following -- 
```python
from data.clothing1m import Clothing1M

dataset = Clothing1M()
dataset.setup()

test_dataloader = dataset.test_dataloader() ## similarly train_dataloader, and val_dataloader can be loaded

for data in test_dataloader:
    images = data['images'] ## data to train on; dimm [batch_size, 3, 256, 256]
    labels = data['labels'] ## target class labels; dimm [batch_size, 1]
    break
```

Checkout this nb for usage on Data Loader -- [nb](https://github.com/askmuhsin/clothing1M_experiments/blob/main/notebooks/test_pl_data_module.ipynb)
