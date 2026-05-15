# xLoc

**Learning to Detect and Localize Multilingual Bugs**

| | |
|---|---|
| Original artifact | <https://figshare.com/s/0441ddd801cddaeb1f2b> |
| Imported from | the publications page |
| Tool | `pubs2github` |


---

## Contents

The artifact contains 1626 file(s) including Python, Config files, and Documentation.

```
├── code
│   ├── ct_vocab.json
│   ├── main_api_call_position.py
│   └── main_codet52.py
├── data
│   └── fine-tune_data
├── models
├── raw_prediction
│   ├── API-aware FT.log
│   ├── baseline.log
│   ├── FCPASP + API-aware FT.log
│   ├── FCPASP.log
│   └── xLoc.log
├── transformers
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── activations.cpython-38.pyc
│   │   ├── activations_tf.cpython-38.pyc
│   │   ├── configuration_utils.cpython-38.pyc
│   │   ├── convert_graph_to_onnx.cpython-38.pyc
│   │   ├── convert_pytorch_checkpoint_to_tf2.cpython-38.pyc
│   │   ├── convert_slow_tokenizer.cpython-38.pyc
│   │   ├── convert_slow_tokenizers_checkpoints_to_fast.cpython-38.pyc
│   │   ├── convert_tf_hub_seq_to_seq_bert_to_pytorch.cpython-38.pyc
│   │   ├── debug_utils.cpython-38.pyc
│   │   ├── deepspeed.cpython-38.pyc
│   │   ├── dependency_versions_check.cpython-38.pyc
│   │   ├── dependency_versions_table.cpython-38.pyc
│   │   ├── dynamic_module_utils.cpython-38.pyc
│   │   ├── feature_extraction_sequence_utils.cpython-38.pyc
│   │   ├── feature_extraction_utils.cpython-38.pyc
│   │   ├── file_utils.cpython-38.pyc
│   │   ├── generation_beam_constraints.cpython-38.pyc
│   │   ├── generation_beam_search.cpython-38.pyc
│   │   ├── generation_flax_logits_process.cpython-38.pyc
│   │   ├── generation_flax_utils.cpython-38.pyc
│   │   ├── generation_logits_process.cpython-38.pyc
│   │   ├── generation_stopping_criteria.cpython-38.pyc
│   │   ├── generation_tf_logits_process.cpython-38.pyc
│   │   ├── generation_tf_utils.cpython-38.pyc
│   │   ├── generation_utils.cpython-38.pyc
│   │   ├── hf_argparser.cpython-38.pyc
│   │   ├── image_utils.cpython-38.pyc
│   │   ├── integrations.cpython-38.pyc
│   │   ├── keras_callbacks.cpython-38.pyc
│   │   ├── modelcard.cpython-38.pyc
│   │   ├── modeling_flax_outputs.cpython-38.pyc
│   │   ├── modeling_flax_pytorch_utils.cpython-38.pyc
│   │   ├── modeling_flax_utils.cpython-38.pyc
│   │   ├── modeling_outputs.cpython-38.pyc
│   │   ├── modeling_tf_outputs.cpython-38.pyc
│   │   ├── modeling_tf_pytorch_utils.cpython-38.pyc
│   │   ├── modeling_tf_utils.cpython-38.pyc
│   │   ├── modeling_utils.cpython-38.pyc
│   │   ├── optimization.cpython-38.pyc
│   │   ├── optimization_tf.cpython-38.pyc
│   │   ├── processing_utils.cpython-38.pyc
│   │   ├── pytorch_utils.cpython-38.pyc
│   │   ├── testing_utils.cpython-38.pyc
│   │   ├── tf_utils.cpython-38.pyc
│   │   … (57 more items)
│   … (1884 more items)
… (1899 more items)
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
