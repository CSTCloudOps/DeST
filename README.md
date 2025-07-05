# DeST

## Quick Started
### Environment
Python 3.9.13, PyTorch 1.12.1, scikit-learn 1.1.2, and DGL 0.9.0 are suggested.


### Dataset: 

In the folder:

- `cases` : There are two files in this directory.

  - `ad_cases.pkl`: The list of failure timestamps.

  - `cases.csv`: The four items in the table header indicate the failure injection time, failure level, root cause of the failure, and failure type respectively.

- `hash_info` : There are four files in this directory. They all hold a dictionary that records the correspondence between names and indexes.

- `samples` : There are three files in this directory, all samples (`samples.pkl`), samples for pre-training (`train_samples.pkl`), and samples for evaluation (`test_samples.pkl`).

Each sample is a tuple: (timestamp, graphs, features of each node). Graphs indicate the topology of the microservice system generated from call relationships and deployment information; Features of each node are composed of pod metric feats, pod trace feats, pod log feats and node metric feats.

### Demo

We provide a demo. Before running the following commands, please unzip D1.zip and D2.zip.

```python
python main.py
```

## Overview


# 🎉 Acknowledgement
We appreciate the following github repos a lot for their valuable code base:

https://github.com/bbyldebb/ART

https://github.com/dawnvince/EasyTSAD
