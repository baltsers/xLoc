# xLoc

**Learning to Detect and Localize Multilingual Bugs**

| | |
|---|---|
| Original artifact | <https://figshare.com/s/f0a7d357b148fc262a40> |
| Imported from | the publications page |
| Tool | `pubs2github` |


## Other papers sharing this artifact

- Generating Realistic Vulnerabilities via Neural Code Editing: An Empirical Study

---

## Contents

The artifact contains 4 file(s), primarily Documentation.

```
├── code_models.zip
├── data.zip
├── exsy_t1_mix_inphase_cs_comb.vt
└── README.md
```

---

## Original `README.md` (from the upstream artifact)

# Learning to Detect and Localize Multilingual Bugs [Artifact]

This is the artifact of the paper *Learning to Detect and Localize Multilingual Bugs*

## 1. Overview
This artifact contains models, data, raw prediction result.

## 2. Directory structure

data - Inside the "data" folder, there is "fine-tune_data", which contains the datasets used for fine-tuning.

models - The "models" folder contains five models, which include "xloc", "baseline", and three ablation version models.

raw_prediction - The "raw_prediction" folder contains the results of the "xloc", "baseline", and three ablation version models on the test set.

transformers - The "transformers" folder contains the code for the xloc model.

code - The "code" folder contains the code we used for pre-training, fine-tuning, and testing various models.

## 3. Usage

For "transformers" folder, users need to copy this folder to the Python installation directory. For example, we need to copy this folder to */opt/conda/lib/python3.8/site-packages/*.

For "code" folder, the *main_api_call_position.py* used for API-aware model. the *main_codet52.py* used for non-API-aware model.
