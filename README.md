ProDiT 
====================================

This repository contains the code for the ProDiT manuscript "A Transformer-Based Dual-Input Framework Improves Protein Mutation Effect Predictions". 

Required Python packages
------------------------

```
pandas
numpy
pyyaml
scikit-learn
torch
matplotlib
```

Data preprocessing
------------------

To run the code, please donwload the data folder from Zenodo (https://zenodo.org/record/15480956) and place the unzipped folder in the main GitHub directory.


1. Rosetta score preprocessing

   ```
   python code/preprocessing/preprocess_rosetta.py \
          --data_path <PATH_TO_DATA> \
          --dataset <DATASET_NAME>
   ```

   Arguments:
   * **data_path** – path to the *data* directory
   * **dataset**   – dataset identifier (avgfp, dlg4-2022-binding, gb1, grb2-abundance, grb2-binding, pab1, tem-1, ube4b)

2. Experimental measurement preprocessing

   ```
   python code/preprocessing/preprocess_experimental_data.py \
          --data_path <PATH_TO_DATA> \
          --dataset <DATASET_NAME>
   ```

   Arguments identical to step 1.

Model pre-training on Rosetta scores
-----------------------------------

```
python code/training/pretrain.py \
       --data_path <PATH_TO_DATA> \
       --dataset <DATASET_NAME> \
       --embedding_size 256 --num_layers 8 --num_heads 8 \
       --feedforward_dim 512 --feedforward_class_dim 64 \
       --lr 0.0001 --weight_decay 0.1 \
       --dropout_prob_head 0.1 --dropout_prob_TN 0.1 \
       --batch_size 128 --num_epochs 10 --val_r 20
```

Available arguments for *pretrain.py*

```
--embedding_size          int   size of token embeddings
--num_layers              int   number of transformer layers
--num_heads               int   attention heads per layer
--feedforward_dim         int   hidden size of feed-forward blocks
--feedforward_class_dim   int   hidden size of prediction head
--dropout_prob_head       float dropout in prediction head
--dropout_prob_TN         float dropout in encoder blocks
--batch_size              int   mini-batch size
--lr                      float learning rate
--weight_decay            float weight-decay coefficient
--num_epochs              int   training epochs
--data_path               str   path to *data* directory
--pretrained_model        str   (optional) checkpoint to resume from
--dataset                 str   dataset identifier
--val_r                   int   reference samples per sequence for validation
```

Supervised training on experimental data
---------------------------------------

```
python code/training/train.py \
       --data_path <PATH_TO_DATA> \
       --dataset <DATASET_NAME> \
       --training_size <50|100|200|500|1000|2000|5000> \
       --embedding_size 256 --num_layers 8 --num_heads 8 \
       --feedforward_dim 512 --feedforward_class_dim 64 \
       --lr 0.0001 --lr_phase1 0.0001 --weight_decay 0.1 \
       --dropout_prob_head 0.1 --dropout_prob_TN 0.1 \
       --batch_size 128 --num_epochs 2000 --phase1_epochs 50 \
       --val_r 20 \
       --pretrained_model <PATH_TO_PRETRAINED_MODEL>
```

Available arguments for *train.py*

```
--embedding_size          int   size of token embeddings
--num_layers              int   number of transformer layers
--num_heads               int   attention heads per layer
--feedforward_dim         int   hidden size of feed-forward blocks
--feedforward_class_dim   int   hidden size of prediction head
--dropout_prob_head       float dropout in prediction head
--dropout_prob_TN         float dropout in encoder blocks
--batch_size              int   mini-batch size
--lr                      float learning rate after encoder unfreeze
--lr_phase1               float learning rate for frozen-encoder phase
--weight_decay            float weight-decay coefficient
--num_epochs              int   total training epochs
--phase1_epochs           int   epochs before encoder unfreezing
--data_path               str   path to *data* directory
--pretrained_model        str   checkpoint from the pre-training step
--dataset                 str   dataset identifier
--val_r                   int   reference samples per sequence for validation
--training_size           int   number of training examples (50–5000)
```

