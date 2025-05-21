import torch
import torch.nn as nn
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


class TransformerEncoder(nn.Module):
    def __init__(self,  output_dim, embedding_size, num_layers, num_heads, feedforward_dim,
             feedforward_class_dim, dropout_head, dropout_TN, vocab_size, seq_length):
        super(TransformerEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.classification_token = nn.Parameter(torch.ones(1, 1, embedding_size))
        self.token_embedding = nn.Embedding(vocab_size, int(embedding_size/2))
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, int(embedding_size/2)))

        self.encoder_layers = nn.ModuleList([
           nn.TransformerEncoderLayer(embedding_size, num_heads, feedforward_dim, dropout_TN)
            for _ in range(num_layers)
        ])

        self.feedforward_classification = nn.Sequential(
            nn.Linear(embedding_size, feedforward_class_dim),
            nn.Dropout(dropout_head),
            nn.ReLU(),
            nn.Linear(feedforward_class_dim, output_dim)
        )
        
    def forward(self, x1, x2):
        cls_tokens = self.classification_token.expand(x1.size(0), -1, -1)

        x1 = self.token_embedding(x1) + self.positional_encoding[:, :x1.size(1), :]
        x2 = self.token_embedding(x2) + self.positional_encoding[:, :x2.size(1), :]
        x1_x2_diff = x1 - x2

        x = torch.cat((x1, x1_x2_diff), dim=2)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x.permute(1, 0, 2)

        for layer in self.encoder_layers:
            x = layer(x)

        #average pooling
        token_final = torch.mean(x, dim=0)
        scores = self.feedforward_classification(token_final)
        return scores
    


def load_pretrained_model(current_model, pretrained_state_dict):
    current_state_dict = current_model.state_dict()
    new_state_dict = {}
    
    for name, param in pretrained_state_dict.items():
        if name in current_state_dict:
            if current_state_dict[name].shape == param.shape:
                new_state_dict[name] = param
            else:
                logging.info(f"Skipping parameter {name} due to shape mismatch.")
        else:
            logging.info(f"Parameter {name} not found in current model.")

    current_model.load_state_dict(new_state_dict, strict=False)
    return current_model
    

def val_model(val_dataloader, model, device, approach = 1, r=20, return_predictions = False):
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for sequences1, sequences2, batch_input_scores, batch_ref_scores in val_dataloader:

            sequences1 = sequences1.to(device)
            sequences2 = sequences2.to(device)

            if approach == 1:
                output = model(sequences1, sequences2)
                output_np = output.cpu().detach().numpy().reshape(-1)
                predictions = output_np + np.array(batch_ref_scores).reshape(-1)
                for k in range(int(len(sequences1)/r)):
                    all_predictions.append(np.mean(predictions[k*r:(k+1)*r]))
                    all_labels.append(np.array(batch_input_scores)[k*r])
                    
            elif approach == 2:
                output = model(sequences2, sequences1)
                output_np = output.cpu().detach().numpy().reshape(-1)
                predictions = (output_np - np.array(batch_ref_scores).reshape(-1)) *(-1)
                for k in range(int(len(sequences1)/r)):
                    all_predictions.append(np.mean(predictions[k*r:(k+1)*r]))
                    all_labels.append(np.array(batch_input_scores)[k*r])
    
    r2 = r2_score(all_labels, all_predictions)
    pearson = pearsonr(all_labels, all_predictions)[0]
    mse = mean_squared_error(all_labels, all_predictions)
    logging.info(f"Approach: {approach:.1f}, R-squared: {r2:.4f}, Pearson: {pearson:.4f}, MSE: {mse:.4f}")

    if return_predictions:
        return r2, all_predictions, all_labels
    else:
        return r2
    

def val_model_pretraining(val_dataloader, model, device, approach = 1, r=20, return_predictions = False):
    all_predictions = []
    all_labels = []

    model.eval()
    print("Validation")
    with torch.no_grad():
        for sequences1, sequences2, batch_input_scores, batch_ref_scores in val_dataloader:

            sequences1 = sequences1.to(device)
            sequences2 = sequences2.to(device)

            if approach == 1:
                output = model(sequences1, sequences2)
                output_np = output.cpu().detach().numpy()
                predictions = output_np + np.array(batch_ref_scores)

                for k in range(int(len(sequences1)/r)):
                    all_predictions.append(np.mean(predictions[k*r:(k+1)*r], axis =0).reshape(-1))
                    all_labels.append(np.array(np.array(batch_input_scores)[k*r, :]).reshape(-1))
            elif approach == 2:
                output = model(sequences2, sequences1)
                output_np = output.cpu().detach().numpy()
                predictions = (output_np - np.array(batch_ref_scores)) *(-1)
                for k in range(int(len(sequences1)/r)):
                    all_predictions.append(np.mean(predictions[k*r:(k+1)*r], axis =0).reshape(-1))
                    all_labels.append(np.array(np.array(batch_input_scores)[k*r, :]).reshape(-1))
    all_labels = np.array(all_labels).reshape(-1)
    all_predictions = np.array(all_predictions).reshape(-1)

    
    r2 = r2_score(all_labels, all_predictions)
    pearson = pearsonr(all_labels, all_predictions)[0]
    mse = mean_squared_error(all_labels, all_predictions)
    logging.info(f"Approach: {approach:.1f}, R-squared: {r2:.4f}, Pearson: {pearson:.4f}, MSE: {mse:.4f}")

    if return_predictions:
        return r2, all_predictions, all_labels
    else:
        return r2