
This repository is for the related paper Meta Operator for Complex Query Answering(see [OpenReview Link](https://openreview.net/forum?id=wfqXYwfmY5B&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FWorkshop%2FMetaLearn%2FAuthors%23your-submissions))).







## Data preparation
The original data can be downloaded from (https://cloud.tsinghua.edu.cn/f/f62f793cdb604737a7eb/?dl=1) and then unzip into the data folder.
It can also be directly derived by running `utils/create_mini_dataset.py`

The structure of the data folder should be like

```
data
	|---FB15k-237-betae-1p-0001
```




## Run code

The detailed setting of hyper-parameters or the knowledge graph to choose are in `config` folder,
you can modify those configurations to create your own, all the experiments are on FB15k-237, retain only 0.1% of data except for one hop 
by default.

Then we can run the few-shot learning experiment by specifying the config.

To run original version of LogicE:
```
python main.py --config config/MAMO/1p-0001_LogicE_onlyP.yaml
```

To run the MAML version:
```bash
python main.py --config config/MAMO/LogicE_task_MAML_1p-0001_onlyP.yaml
```

To run our MAMO version, run the following, the classification principle is specified.
```

python main.py --config config/MAMO/LogicE_0001_onlyP_leaf_lr_0.008.yaml
python main.py --config config/MAMO/LogicE_0001_onlyP_root_lr_0.008.yaml
python main.py --config config/MAMO/LogicE_0001_onlyP_output_lr_0.008.yaml
python main.py --config config/MAMO/LogicE_0001_onlyP_input_lr_0.008.yaml
```


