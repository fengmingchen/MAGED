# MAGED:Multimodal Attentive Graph learning with Gene Expression Dynamics on Knowledge Graphs for TCM Target Prediction
This repo contains a PyTorch implementation for MAGED , which is MAGED model proposed in our paper **"Multimodal Attentive Graph learning with Gene Expression Dynamics on Knowledge Graphs for TCM Target Prediction"**.
## Overview 
We propose an end-to-end multimodal learning framework based on a Heterogeneous Graph Attention Network (HGAT), which formulates Herb–Target Interaction (HTI) prediction as a knowledge graph link prediction task. Our model integrates multimodal features of herbs, including text-based attribute embeddings from pre-trained language models and pathway enrichment scores (NES) from transcriptomic data, to construct dynamic contextual representations. It employs a hierarchical, context-aware graph attention mechanism to enhance node representations and suppress noisy connections. The framework finally predicts unknown herb–target associations via Dot scoring function. Our model effectively combines molecular biological mechanisms with complex network topology, enabling more accurate and robust HTI prediction. Moreover, the incorporation of a dynamic contextual gating mechanism significantly enhances the model's adaptability and generalization capability in noisy environments.


<img width="800" height="703" alt="abstract" src="https://github.com/user-attachments/assets/b5428f41-592d-4c58-9a8c-6944ada04a77" />

## Requirements
To run our code, following main dependency packages are needed:\
Package	Version
```
python  3.8
cupy-cuda12x	12.3.0
matplotlib	3.7.5
networkx	3.1
numpy	1.24.4
pandas	1.5.2
torch	2.4.1
torch-geometric	2.6.1
torch_scatter	2.1.2
tqdm	4.67.1
scikit-learn	1.3.2
sentence-transformers	3.2.1
```
## Usage
### Data & Code Prepare
To run MAGED, please clone the repository and extract the data files from the archive in the `data` directory. Additionally, this project uses the Chinese text-encoding model `[BAAI/bge-small-zh-v1.5]`. To ensure operation in offline or intranet environments, please download the model to the local directory `MAGED\map_context_file\bge-small-zh-v1.5`. Model homepage (for reference): https://huggingface.co/BAAI/bge-small-zh-v1.5. You will need paths for:
- `--property_csv_path` : Herb attribute table (e.g., 四气五味/归经/功效词), CSV.
- `--nes_csv_path`      : NES matrix CSV.
- `--mapping_csv_path`  : Herb Chinese name ↔ `TCM_ID` mapping CSV.
- `--data_dir`          : KG triples directory for `KnowledgeGraphBuilder`.
- `--text_model`        : Local sentence-transformers directory (e.g., `bge-small-zh-v1.5`).\
Output directories:
- `--results_dir` : All training curves, fold metrics, and cache will go here.
- `--model_dir`   : When `--save_model` is set (train_test), best fold checkpoints + entity maps are saved here.
### Training Stage
To train MAGED, use the following commands with the default paths, or customize the training by replacing the file path arguments. Run from project root (where the `MAGED/` folder lives). Include the `--save_model` flag to save the best model from each fold and the entity mappings:
```
cd MAGED/
python -m src.train_test --save_model
```
Alternatively, specify custom paths with:
```
cd MAGED/
python -m src.train_test ^
  --property_csv_path "...\herb_properties.csv" ^
  --nes_csv_path      "...\herb_nes_matrix.csv" ^
  --mapping_csv_path  "...\herb_map_id.csv" ^
  --data_dir          "...\data" ^
  --results_dir       "...\results" ^
  --text_model        "...\bge-small-zh-v1.5" ^
  --save_model
```
### Inference Stage
During the inference phase, users can leverage the saved models from the training step to evaluate test metrics and/or export the Top-K predicted targets for a given herb. To run inference with default paths:
```
python -m src.infer --topk_herb "黄芩" --topk_k 300
```
If custom input data directories were used during training, ensure consistency by specifying the same paths during inference:
```
python -m src.infer ^
  --property_csv_path "......\herb_properties.csv" ^
  --nes_csv_path      "......\herb_nes_matrix.csv" ^
  --mapping_csv_path  "......\herb_map_id.csv" ^
  --data_dir          "......\data" ^
  --results_dir       "......\results" ^
  --text_model        "......\bge-small-zh-v1.5" ^
  --topk_herb         "黄芩" ^
  --topk_k            300
```
## Contact
17801111879@163.com
