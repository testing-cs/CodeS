# CodeS-distribution-shift-benchmark-datasets

All the datasets and models can be downloaded at [figshare](https://figshare.com/s/16e923c6d4d94e3559ba).

**Datasets: Python75.zip, Java250.zip, Python800.zip**

Three collections of datasets: Python75, Java250-S, Python800-S. Each collection has the same structure of directories. For example:
Python75.zip:


    ├── raw                           # Raw data files scrapped from the online resources.
    │   ├── [task_name]               # Code files in each task
    │   │   └──  [submission_id].py   # Source code file with the submission id.
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

**Detectors: Detection.zip**: implementations of OOD detectors
How to use:
1. git clone https://github.com/IBM/Project_CodeNet.git
2. Unzip the Detection.zip and put it to Project_CodeNet/tree/main/model-experiments/token-based-similarity-classification/src/
