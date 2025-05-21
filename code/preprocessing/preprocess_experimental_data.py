import argparse
import logging
import os
from os.path import join
import pandas as pd
import yaml
import random
from util import generate_mutant_sequence

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default = "",
        help="Path to data directory of ProDiT",
        )
    parser.add_argument(
        "--dataset",
        type=str,
        default = "",
        help="Name of the dataset that shall be preprocessed",
        )
    return parser.parse_args()

args = get_arguments()

log_output_dir = join(args.data_path, "..", "log_outputs", "preprocessing_experimental")
if not os.path.exists(log_output_dir):
    os.makedirs(log_output_dir)


log_name = join(log_output_dir, "preprocessing_experimental_" + args.dataset + ".log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename=log_name, filemode='a', format='%(levelname)s:%(message)s', level=logging.INFO)
logging.info("Preprocessing experimental data for dataset: " + args.dataset)

#Load Amino Acid Sequence of Wildtype
with open(join(args.data_path ,'all_datasets.yml')) as file:
    all_datasets = yaml.load(file, Loader=yaml.FullLoader)
aa_wt = all_datasets[args.dataset]["wt_aa"]
logging.info("Wildtype amino acid: " + aa_wt)

df = pd.read_csv(join(args.data_path, args.dataset, "experimental", args.dataset + ".tsv"), sep = "\t")
logging.info("Length of dataset: " + str(len(df)))
df["sequence"] = df.apply(lambda row: generate_mutant_sequence(aa_wt, row["variant"]), axis=1)

random.seed(42)
validation_indices = random.sample(range(len(df)), 1000)
test_indices = random.sample(range(len(df)), 2000)
df["set"] = "train"
df.loc[validation_indices, "set"] = "validation"
df.loc[test_indices, "set"] = "test"
df["index"] = df.index
df.to_csv(join(args.data_path, args.dataset, "experimental", args.dataset + "_with_sequence.tsv"), sep = "\t", index = False)

val_df = df[df["set"] == "validation"]
test_df = df[df["set"] == "test"]


for train_set_size in [50,100, 200, 500, 1000, 2000, 5000]:
    train_df = df[df["set"] == "train"]
    sample_size = min(train_set_size, len(train_df))
    logging.info("Creating dataset with " + str(sample_size) + " training samples")
    train_df = train_df.sample(n = sample_size)
    save_path = join(args.data_path, args.dataset, "experimental", str(train_set_size))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_df['index'].to_csv(join(save_path, 'train.txt'), index = False, header = False)
    val_df['index'].to_csv(join(save_path, 'validation.txt'), index = False, header = False)
    test_df['index'].to_csv(join(save_path, 'test.txt'), index = False, header = False)


