# CodeS-distribution-shift-benchmark-datasets

This project is for the paper ``CodeS: A Distribution Shift Benchmark Dataset for Source Code Learning``.
## Datasets and Models

All the datasets and models can be downloaded at [figshare](https://figshare.com/s/16e923c6d4d94e3559ba).

**Datasets: Python75.zip, Java250.zip, Python800.zip**

Each collection of dataset has the same structure of directories. Take Python75.zip as an example:


    ├── raw                           # Raw data files scrapped from the online resources.
    │   ├── [task_name]               # Code files in each task
    │   │   └──  [submission_id].py   # Source code file (*.py) with the submission id.
    │   ├── csv                       # Data descriptions
    │   │   └──  [task_name].csv      # The description (e.g., Submission_id, Task_name, User) of each code file for this task
    ├── task                          # Datasets with the task distribution shift 
    │   ├── pre-trained               # Data files for pre-trained language models
    │   │   ├── train.jsonl           # Code files and labels in the trianing set
    │   │   ├── id_test.jsonl         # Code files and labels in the ID test set
    │   │   ├── ood_test.jsonl        # Code files and labels in the OOD test set
    │   ├── token                     # Data files for DNN models
    │   │   ├── train                 # Training set
    │   │   │   ├── [task_name].tkn   # Token representations of source code files in this task for training
    │   │   │   ├── info.json         # Information of the programming language and number of tokens
    │   │   │   └── problems.json     # Information of the data size of each task
    │   │   ├── id_test               # ID test set
    │   │   │   ├── [task_name].tkn   # Token representations of source code files in this task for ID test
    │   │   │   ├── info.json         # Information of the programming language and number of tokens
    │   │   │   └── problems.json     # Information of the data size of each task
    │   │   ├── ood_test              # OOD test set
    │   │   │   ├── [task_name].tkn   # Token representations of source code files in this task for OOD test
    │   │   │   ├── info.json         # Information of the programming language and number of tokens
    │   │   │   └── problems.json     # Information of the data size of each task
    ├── user                          # Datasets with the programmer distribution shift 
    │   └── ...                       # The same as task structure
    ├── time                          # Datasets with the time distribution shift 
    │   └── ...                       # The same as task structure
    ├── token                         # Datasets with the token distribution shift 
    │   └── ...                       # The same as task structure
    ├── cst                           # Datasets with the cst distribution shift 
    │   └── ...                       # The same as task structure
    ├── random                        # Datasets with no distribution shift 
    │   └── ...                       # The same as task structure

**Models: models.zip**: trained models and OE detectors

    ├── cnns                                                                  # Trained DNNs (CNN(sequence) and MLP(Bag)) using the training and ID test sets with different distribution shifts.
    │   └── [DNN name]-[data name]-[distribution shift type].h5               # Trained DNN with a specific architecture, for a specific dataset with a certain distribution shift
    ├── oe_detectors                                                          # Trained OE detectors
    │   └── [DNN name]-[data name]-[distribution shift type]-oe.h5            # OE detector with a specific architecture, for a specific dataset with a certain distribution shift

## OOD detectors
The implementation of 4 OOD detectors are under the directory ``Detection/``.

```
mspDetector.py                         # The implementation of the Maximum Softmax Probability (MSP) detector.
odinDetector.py                        # The implementation of the Out-of-Distribution detector for neural networks (ODIN) detector.
mahalanobisDetector.py                 # The implementation of the Mahalanobis detector.
oeDetector.py                          # Implementation of the Outlier Exposure (OE) detector.
``` 

To obtain the AUROC of OOD detectors. Run:
 ```
python Detection/evaluation.py --data_name java250 --result_dir [user define] --metric random --detector odin
 ```
This command calculates the AUROC of the ODIN detector for the java250 dataset with random distribution shift.


How to use the OOD detectors:
1. git clone https://github.com/IBM/Project_CodeNet.git
2. put the ``Detection`` directory into Project_CodeNet/tree/main/model-experiments/token-based-similarity-classification/src/
3. run the commands to obtain the AUROC scores.

## Acknowledgement

We appreciate the authors, Puri et al., of the [Project CodeNet](https://github.com/IBM/Project_CodeNet) for making their datasets and code publicly available. The raw source code files in ``Java250.zip`` and ``Python800.zip`` are from CodeNet. We also tokenize source code files and build the models using the code in CodeNet.

We appreciate the authors, Liang et al., of [ODIN](https://github.com/facebookresearch/odin) for making their code publicly available. We create the ``odinDetector.py`` on the top of this open source code.

We appreciate the authors, Lee et al., of [Mahalanobis](https://github.com/pokaxpoka/deep_Mahalanobis_detector) for making their code publicly available. We create the ``mspDetector.py`` and ``mahalanobisDector.py`` on the top of this open source code.


## Support and maintenance
Feel free to contact Qiang Hu (qiang.hu@uni.lu) and Yuejun Guo (yuejun.guo@uni.lu) if you have further questions. 

This project aims to facilitate the research of distribution shift in source code understanding and we welcome your contributions! Please submit an issue or a pull request and we will try our best to respond in a timely manner. 

## License
This project is under the [MIT license](https://github.com/testing-cs/CodeS/blob/main/LICENSE).

The raw source code files in ``Java250.zip`` and ``Python800.zip`` come from the [Project CodeNet](https://github.com/IBM/Project_CodeNet) under the [Apache License 2.0 license](https://github.com/IBM/Project_CodeNet/blob/main/LICENSE).

We manually scrape the source code files in ``Python75.zip`` from [AtCoder](https://atcoder.jp/), a public programming contest site. 
**!Note**: we only scrape public-facing data and respect the [Privacy Policy](https://atcoder.jp/privacy) and [Copyright](https://atcoder.jp/tos) declared by AtCoder.

