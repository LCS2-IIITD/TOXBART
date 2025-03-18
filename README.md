# TOXBART

This is the official repository for the paper "[Tox-BART: Leveraging Toxicity Attributes for Explanation Generation of Implicit Hate Speech](https://arxiv.org/abs/2406.03953)."

---

### Setup
Run the following command to initialize the directory:
```bash
git clone https://github.com/LCS2-IIITD/TOXBART.git
cd TOXBART
```
We suggest using `conda` to manage virtual environments and ease-of-access.

#### Strong requirements
```
-> Python >=3.9
-> PyTorch >=2.0
-> Transformers
```

#### Data prerequisites
Kindly go through the following instructions to download the required datasets:
- **[SBIC](https://maartensap.com/social-bias-frames/index.html)**: Need to download from the attached link and place in the `data` directory.
- **[LatentHatred](https://github.com/SALT-NLP/implicit-hate)**: Need to download from the attached link and place in the `data` directory, in addition to the already present data.
- **[Jigsaw Unintended Bias](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)**: Need to download from the attached link and place in the `tox_bert/jigsaw-data` directory.
- **[StereoKG](https://github.com/uds-lsv/StereoKG)**: Need to download from the attached link and place in the `data` directory.
- **ConceptNet**: Use the following command `wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz` in the `data` directory. To unzip the file, use `gunzip conceptnet-assertions-5.7.0.csv.gz`.

---

### Training and Inference
To train the `toxic-bert` model, run the `train.py` file in the `tox_bert` directory. You can also run an inference using `test.py` to see how well it performs in terms of the RMSE scores.

Kindly follow the running instructions present in each `{train, test}.py` file to run the knowledge or configuration experiments.

---

### Directory Structure
Although the code uses relative paths and in most cases asks the user to enter the paths (using `argparse`), we encourage you to not add any depth to the current directory structure. In case, you encounter any issues with the paths, cross-check if your data is in the correct directory.

Here is a an approximate view of how the directories are structured in this repository.

```
.
└── TOXBART/
    ├── data/
    ├── kg_exps/
    │   ├── knowledge_utils.py
    │   ├── stereokg_utils.py
    │   ├── utils.py
    │   ├── test.py
    │   └── train.py
    ├── tox_bert/
    │   ├── jigsaw_data/
    │   ├── modeling_toxbert.py
    │   ├── test.py
    │   └── train.py
    └── toxic_signals/
        ├── config1/
        │   ├── test.py
        │   └── train.py
        ├── config2/
        │   ├── test.py
        │   └── train.py    
        ├── config3/
        │   ├── modeling_toxbart.py
        │   ├── test.py
        │   └── train.py
        ├── config4/
        │   ├── modeling_toxbart.py
        │   ├── test.py
        │   └── train.py
        └── config5/
            ├── modeling_toxbart.py
            ├── test.py
            └── train.py
```

---

### Citing our work
Kindly use the following bibtex to cite our work, thank you.
```bib
@inproceedings{yadav-etal-2024-tox,
    title = "Tox-{BART}: Leveraging Toxicity Attributes for Explanation Generation of Implicit Hate Speech",
    author = "Yadav, Neemesh  and
      Masud, Sarah  and
      Goyal, Vikram  and
      Akhtar, Md Shad  and
      Chakraborty, Tanmoy",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.831/",
    doi = "10.18653/v1/2024.findings-acl.831",
    pages = "13967--13983",
    abstract = "Employing language models to generate explanations for an incoming implicit hate post is an active area of research. The explanation is intended to make explicit the underlying stereotype and aid content moderators. The training often combines top-k relevant knowledge graph (KG) tuples to provide world knowledge and improve performance on standard metrics. Interestingly, our study presents conflicting evidence for the role of the quality of KG tuples in generating implicit explanations. Consequently, simpler models incorporating external toxicity signals outperform KG-infused models. Compared to the KG-based setup, we observe a comparable performance for SBIC (LatentHatred) datasets with a performance variation of +0.44 (+0.49), +1.83 (-1.56), and -4.59 (+0.77) in BLEU, ROUGE-L, and BERTScore. Further human evaluation and error analysis reveal that our proposed setup produces more precise explanations than zero-shot GPT-3.5, highlighting the intricate nature of the task."
}
```

---

Thank you for using our repository. If you face any issues, kindly raise an issue or contact any of the primary authors directly!
