import argparse
import logging
import os
from os.path import join
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util_model import TransformerEncoder, load_pretrained_model, val_model_pretraining
from util_data import ProteinDataset, generate_val_data, plot_r2s


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding_size",
        type=int,
        default = 256,
        help="Size of the embedding",
        )
    parser.add_argument(
        "--num_layers",
        type=int,
        default = 8,
        help="Number of transformer layers",
        )
    parser.add_argument(
        "--num_heads",
        type=int,
        default = 8,
        help="Number of attention heads",
        )
    parser.add_argument(
        "--feedforward_dim",
        type=int,
        default = 512,
        help="Dimension of the feedforward network",
        )
    parser.add_argument(
        "--feedforward_class_dim",
        type=int,
        default = 64,
        help="Dimension of the feedforward network for classification",
        )
    parser.add_argument(
        "--dropout_prob_head",
        type=float,
        default = 0.1,
        help="Dropout probability for the prediction head",
        )
    parser.add_argument(
        "--dropout_prob_TN",
        type=float,
        default = 0.1,
        help="Dropout probability for the encoder layers",
        )
    parser.add_argument(
        "--batch_size",
        type=int,
        default = 128,
        help="Batch size",
        )
    parser.add_argument(
        "--lr",
        type=float,
        default = 1e-4,
        help="Learning rate",
        )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default = 0.1,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default = 1000,
        help="Number of epochs",
        )
    parser.add_argument(
        "--data_path",
        type=str,
        default = "",
        help="Path to data directory of ProDiT",
        )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default = "",
        help="Path to the pre-trained model",
        )
    parser.add_argument(
        "--dataset",
        type=str,
        default = "",
        help="Name of the dataset that shall be preprocessed",
        )
    parser.add_argument(
        "--val_r",
        type=int,
        default = 20,
        help="Number of validation refernce samples per sequence",
    )
    return parser.parse_args()

args = get_arguments()

log_output_dir = join(args.data_path, "..", "log_outputs", "pretraining_rosetta")
if not os.path.exists(log_output_dir):
    os.makedirs(log_output_dir)
model_path = join(args.data_path, args.dataset, "rosetta", "models")
if not os.path.exists(model_path):
    os.makedirs(model_path)

setting = "pretraining_rosetta_" + args.dataset + "_lr" + str(args.lr) + "_bs" + str(args.batch_size)
log_name = join(log_output_dir, setting + ".log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename=log_name, filemode='a', format='%(levelname)s:%(message)s', level=logging.INFO)

logging.info('Logging file setup successful! \n')

# Checking if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')


######### Data Preprocessing #########
df = pd.read_csv(join(args.data_path, args.dataset, "rosetta", args.dataset + "_all_scores.csv"), sep = "\t")
logging.info("Length of dataset: " + str(len(df)))

all_seqs = df["sequences"].tolist()
all_scores = []
for i in range(len(df)):
    scores = df["scores"][i]
    scores = np.fromstring(scores[1:-1], sep = " ")
    all_scores.append(scores)
all_scores = np.array(all_scores)
all_scores = all_scores[:, ~np.isnan(all_scores).any(axis=0)]

#2000 val sequences, rest training:
val_seqs = all_seqs[:2000]
val_scores = all_scores[:2000]
train_seqs = all_seqs[2000:]
train_scores = all_scores[2000:]

del all_seqs, all_scores


########### Model Setup ###########
model = TransformerEncoder(
    output_dim=train_scores.shape[1],
    embedding_size=args.embedding_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    feedforward_dim=args.feedforward_dim,
    feedforward_class_dim=args.feedforward_class_dim,
    dropout_head=args.dropout_prob_head,
    dropout_TN=args.dropout_prob_TN,
    vocab_size=21,
    seq_length=len(train_seqs[0]) +1)

model = model.to(device)
logging.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

if args.pretrained_model != "":
    model = load_pretrained_model(model, torch.load(args.pretrained_model))
    logging.info(f'Model loaded from {args.pretrained_model}')


########### Training the model ###########
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

n_batches = max(len(train_scores) // args.batch_size, 1)
dataset = ProteinDataset(train_seqs, train_scores)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

valloader = generate_val_data(input_seqs = val_seqs, ref_seqs = train_seqs,
                               input_scores = val_scores, ref_scores = train_scores, r = args.val_r)


train_r2s, val_r2s_approach1, val_r2s_approach2 = [], [], []
best_val_r2 = -1

for epoch in range(args.num_epochs):
    model.train()
    epoch_loss = 0.0
    count = 0
    all_outputs, all_true_values = np.array([]), np.array([])    

    for sequences1, sequences2, batch_input_scores in dataloader:
        count += args.batch_size
        sequences1 = sequences1.to(device)
        sequences2 = sequences2.to(device)

        batch_input_scores = batch_input_scores.to(device)

        
        optimizer.zero_grad()
        output = model(sequences1, sequences2)

        loss = criterion(output.view(-1) , batch_input_scores.view(-1))
        loss.backward()
        optimizer.step()
        
        output_np = output.cpu().detach().numpy().reshape(-1)
        batch_input_scores_np = batch_input_scores.cpu().detach().numpy().reshape(-1)
        
        all_outputs = np.append(all_outputs, output_np)
        all_true_values = np.append(all_true_values, batch_input_scores_np)
        
        epoch_loss += loss.item()

        if count > 10**6:
            model.eval()
            count = 0
            logging.info("Approach1:")
            val_model_pretraining(val_dataloader = valloader, model = model, device = device, approach = 1, r = args.val_r)
            logging.info("Approach2:")
            val_model_pretraining(val_dataloader = valloader, model = model, device = device, approach = 2, r = args.val_r)
            model.train()

            #only keep last 10**5 entries from all_outputs and all_true_values
            all_outputs = all_outputs[-10**5:]
            all_true_values = all_true_values[-10**5:]

    mse = mean_squared_error(all_true_values, all_outputs)
    r2 = r2_score(all_true_values, all_outputs)
    rmse = np.sqrt(mse)
    train_r2s.append(r2)
    epoch_loss /= n_batches
    logging.info(" ")
    logging.info(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {epoch_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}")


    logging.info("Validation:")
    model.eval()
    val_r2_approach1 = val_model_pretraining(val_dataloader = valloader, model = model, device = device, approach = 1, r = args.val_r)
    val_r2_approach2 = val_model_pretraining(val_dataloader = valloader, model = model, device = device, approach = 2, r = args.val_r)

    val_r2 = max(val_r2_approach1, val_r2_approach2)

    val_r2s_approach1.append(val_r2_approach1), val_r2s_approach2.append(val_r2_approach2)

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        torch.save(model.state_dict(), join(model_path, setting + '_model_checkpoint.pth'))
        logging.info('Model saved successfully\n')
    
    plot_r2s(train_r2s, val_r2s_approach1, val_r2s_approach2, setting, log_output_dir)

logging.info('Training completed successfully!')