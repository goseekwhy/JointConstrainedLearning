# Joint Constrained Learning for Event-Event Relation Extraction

This is the repository for the resources in EMNLP 2020 Paper "Joint Constrained Learning for Event-Event Relation Extraction". This repository contains the source code and datasets used in our paper.

## Abstract

Understanding  natural  language  involves  recognizing  how  multiple  event  mentions  structurally and temporally interact with each other. In  this  process,  one  can  induce  event  complexes that organize multi-granular events with temporal  order  and  membership  relations  interweaving  among  them.   Due  to the  lack  of jointly  labeled  data  for  these  relational  phenomena  and  the  restriction  on  the  structures they articulate, we propose a joint constrained learning framework for modeling event-event relations. Specifically, the framework enforces logical constraints within and across multiple temporal and subevent relations by converting these  constraints  into  differentiable  learning objectives. We show that our joint constrained learning approach effectively compensates for the  lack  of  jointly  labeled  data,  and  outperforms SOTA methods on benchmarks for both temporal relation extraction and event hierarchy construction, replacing a commonly used but  more  expensive  global  inference  process. We also present a promising case study showing the effectiveness of our approach in inducing event complexes on an external corpus.

## How to run the code
`
git clone https://github.com/why2011btv/JointConstrainedLearning.git
conda env create -n conda-env -f environment.yml
pip install requirements.txt

mkdir model_params
cd model_params
mkdir HiEve_best
mkdir MATRES_best
cd ..
`
### Running experiments 
`python3 main.py gpu_0 batch_16 0.0000001 0920_0.rst epoch_40 <SETTING> <LOSS> <FINETUNE>`
SETTING: choose from "MATRES", "HiEve", "Joint"
LOSS: choose from "add_loss_0", "add_loss_1"
FINETUNE: choose from "finetune_0", "finetune_1"

## Reference
`@inproceedings{WCZR20,
    author = {Haoyu Wang and Muhao Chen and Hongming Zhang and Dan Roth},
    title = {{Joint Constrained Learning for Event-Event Relation Extraction}},
    booktitle = {Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2020},
    url = "https://cogcomp.seas.upenn.edu/papers/WCZR20.pdf",
    funding = {KAIROS},
}`