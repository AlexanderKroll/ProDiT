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

from util_model import TransformerEncoder, load_pretrained_model, val_model
from util_data import ProteinDataset, generate_val_data, plot_r2s, load_experimental_data


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
        "--lr_phase1",
        type=float,
        default = 1e-4,
        help="Learning rate for the first phase",
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
        "--phase1_epochs",
        type=int,
        default = 50,
        help="Number of epochs for the first phase",
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
    parser.add_argument(
        "--training_size",
        type=int,
        default = 50,
        help="Size of the training set",
    )
    return parser.parse_args()

args = get_arguments()

log_output_dir = join(args.data_path, "..", "log_outputs", "training_experimental")
if not os.path.exists(log_output_dir):
    os.makedirs(log_output_dir)

model_path = join(args.data_path, args.dataset, "experimental", "models")
if not os.path.exists(model_path):
    os.makedirs(model_path)

prediction_path = join(args.data_path, args.dataset, "experimental", "predictions", str(args.training_size))
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)

setting = "training_" + args.dataset + "_lr" + str(args.lr) + "_bs" + str(args.batch_size) + "_training_size" + str(args.training_size)
log_name = join(log_output_dir, setting + ".log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename=log_name, filemode='a', format='%(levelname)s:%(message)s', level=logging.INFO)

for arg in vars(args):
    logging.info(f'{arg}: {getattr(args, arg)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

########## Load data ##########
df = pd.read_csv(join(args.data_path, args.dataset, "experimental", args.dataset + "_with_sequence.tsv"), sep = "\t")

train_seqs, train_scores = load_experimental_data(df, args.training_size, "train", args.data_path, args.dataset)
val_seqs, val_scores = load_experimental_data(df, args.training_size, "validation", args.data_path, args.dataset)
test_seqs, test_scores = load_experimental_data(df, args.training_size, "test", args.data_path, args.dataset)


train_mean = np.mean(train_scores)
train_std = np.std(train_scores)

train_scores = (train_scores - train_mean) / train_std
val_scores = (val_scores - train_mean) / train_std
test_scores = (test_scores - train_mean) / train_std


logging.info("Length of training dataset: " + str(len(train_seqs)))
logging.info("Length of validation dataset: " + str(len(val_seqs)))
logging.info("Length of test dataset: " + str(len(test_seqs)))

train_dataset = ProteinDataset(train_seqs, train_scores)
dataloader = DataLoader(train_dataset, batch_size = args.batch_size , shuffle=True)
n_batches = int(np.ceil(len(train_scores) / args.batch_size))

valloader = generate_val_data(input_seqs = val_seqs, ref_seqs = train_seqs,
                               input_scores = val_scores, ref_scores = train_scores, r = args.val_r)

testloader = generate_val_data(input_seqs = test_seqs, ref_seqs = train_seqs,
                               input_scores = test_scores, ref_scores = train_scores, r = args.val_r)

########### Model Setup ###########
model = TransformerEncoder(
    output_dim=1,
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


#Freeze transformer parameters for the first 50 epochs
transformer_params, prediction_head_params  = [], []
for name, param in model.named_parameters():
    if 'encoder' in name or "pooling" in name or "classification_token" in name: 
        transformer_params.append(param)
        param.requires_grad = False
    else:
        prediction_head_params.append(param)

logging.info("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        logging.info(name)


########### Training the model ###########
criterion = nn.MSELoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_phase1, weight_decay=args.weight_decay)         

train_r2s, val_r2s_approach1, val_r2s_approach2 = [], [], []

try:
    with open(join(prediction_path, "overall_best_val_r2.txt"), "r") as file:
        overall_best_val_r2 = float(file.read())
except:
    overall_best_val_r2 = -1

best_val_r2 = -1
best_val_epoch = 0
new_lr = args.lr


for epoch in range(args.num_epochs):
    model.train()
    epoch_loss = 0.0


    # Unfreeze transformer parameters after args.phase1_epochs
    if epoch == args.phase1_epochs:
        for param in transformer_params:
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #if val loss did not improve for 50 epochs, decrease learning rate:
    if (epoch - best_val_epoch) > 50:
        new_lr = new_lr/2.0
        best_val_epoch = epoch
        if new_lr < 5*1e-6:
            new_lr = args.lr
        logging.info("Val loss did not improve for 50 epochs, decrease learning rate. New lr: %s" % (new_lr))
        optimizer = optim.AdamW(model.parameters(), lr=new_lr, weight_decay=args.weight_decay)


    all_outputs, all_true_values = np.array([]), np.array([])    
    for sequences1, sequences2, batch_input_scores in dataloader:
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


    mse = mean_squared_error(all_true_values, all_outputs)
    r2 = r2_score(all_true_values, all_outputs)
    rmse = np.sqrt(mse)
    train_r2s.append(r2)
    epoch_loss /= n_batches
    logging.info(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {epoch_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}")


    logging.info("Validation:")
    val_r2_approach1 = val_model(val_dataloader = valloader, model = model, device = device, approach = 1, r = args.val_r)
    val_r2_approach2 = val_model(val_dataloader = valloader, model = model, device = device, approach = 2, r = args.val_r)

    val_r2 = max(val_r2_approach1, val_r2_approach2)

    val_r2s_approach1.append(val_r2_approach1), val_r2s_approach2.append(val_r2_approach2)

    if val_r2 > best_val_r2:
        best_val_r2, best_val_epoch = val_r2, epoch
        


        best_approach = 1 if val_r2_approach1 > val_r2_approach2 else 2

        logging.info("Testing:")
        best_test_r2, test_pred, test_true = val_model(val_dataloader=testloader, model=model, device=device, approach=best_approach,
                                                        r=args.val_r, return_predictions=True)

        if best_val_r2 > overall_best_val_r2:
            overall_best_val_r2 = best_val_r2
            with open(join(prediction_path, "overall_best_val_r2.txt"), "w") as file:
                file.write(str(overall_best_val_r2))

            np.save(join(prediction_path, 'test_predictions.npy'), test_pred)
            np.save(join(prediction_path, 'test_true.npy'), test_true)

            torch.save(model.state_dict(), join(model_path, setting + '_model_checkpoint.pth'))
            logging.info('Model saved successfully\n')

        logging.info(f"Best test R-squared: {best_test_r2}")
        logging.info("\n")
        
    plot_r2s(train_r2s, val_r2s_approach1, val_r2s_approach2, setting, log_output_dir)

logging.info('Training completed successfully!')