from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from os.path import join
import matplotlib.pyplot as plt

vocab = {'A': 0, 'Q': 1, 'Y': 2, 'V': 3, 'G': 4, 'F': 5, 'K': 6, 'L': 7, 'M': 8, 'E': 9, 'C': 10,
         'I': 11, 'P': 12, 'N': 13, 'W': 14, 'D': 15, 'H': 16, 'R': 17, 'T': 18, 'S': 19, '<UNK>': 20}

def tokenize(text):
    return np.array([vocab.get(word, vocab["<UNK>"]) for word in text])

class ProteinDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        random_ind = np.random.randint(len(self.sequences))
        random_seq = self.sequences[random_ind]
        random_score = self.scores[random_ind]
        return tokenize(self.sequences[idx]), tokenize(random_seq), np.array(self.scores[idx] - random_score, dtype=np.float32)
    

class Val_ProteinDataset(Dataset):
    def __init__(self, sequences1, sequences2, true_scores, ref_scores):
        self.sequences1 = sequences1
        self.sequences2 = sequences2
        self.true_scores = true_scores
        self.ref_scores = ref_scores

    def __len__(self):
        return len(self.sequences1)

    def __getitem__(self, idx):
        return(tokenize(self.sequences1[idx]), tokenize(self.sequences2[idx]), np.array(self.true_scores[idx], dtype=np.float32), np.array(self.ref_scores[idx], dtype=np.float32))


def generate_val_data(input_seqs, ref_seqs, input_scores, ref_scores, r = 20, batch_size = 120):
    all_input_sequences = []
    all_ref_input_sequences = []
    all_ref_y = []
    all_true_y = []
    for i in range(len(input_seqs)):
        for _ in range(r):
            ind = np.random.randint(0, len(ref_seqs))
            all_input_sequences.append(input_seqs[i])
            all_ref_input_sequences.append(ref_seqs[ind])
            all_ref_y.append(ref_scores[ind])
            all_true_y.append(input_scores[i])

    logging.info("Number of validation samples: %s" % (len(all_input_sequences)))
    val_dataset = Val_ProteinDataset(all_input_sequences, all_ref_input_sequences,  all_true_y, all_ref_y)
    dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
    return dataloader

def plot_r2s(train_r2s, val_r2s_approach1, val_r2s_approach2, setting, log_output_dir):
    plt.plot(np.array(train_r2s), label='train')
    plt.plot(np.array(val_r2s_approach1), label='val_approach1')
    plt.plot(np.array(val_r2s_approach2), label='val_approach2')
    plt.ylim(max(0,min(min(val_r2s_approach1), min(val_r2s_approach2), min(train_r2s)) ), max(max(val_r2s_approach1), max(val_r2s_approach2), max(train_r2s))+0.05)
    plt.legend()
    plt.savefig(join(log_output_dir,  setting + '_r2_plot.png'))
    plt.close()


def load_experimental_data(df, n, set_name, data_path, dataset):
    with open(join(data_path, dataset, "experimental", str(n), set_name + ".txt")) as f:
        lines = f.readlines()

    ind = [int(ind.replace("\n", "")) for ind in lines]
    seqs = [df.iloc[i]["sequence"] for i in ind]
    scores = [df.iloc[i]["score"] for i in ind]
    return seqs, scores