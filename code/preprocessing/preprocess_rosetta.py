import argparse
import logging
import os
from os.path import join
import pandas as pd
import numpy as np
import yaml
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

log_output_dir = join(args.data_path, "..", "log_outputs", "preprocessing_rosetta")
if not os.path.exists(log_output_dir):
    os.makedirs(log_output_dir)

#initialize logger
log_name = join(log_output_dir, "preprocessing_rosetta_" + args.dataset + ".log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename=log_name, filemode='a', format='%(levelname)s:%(message)s', level=logging.INFO)
logging.info('Logging file setup successful! \n')

# Load dataset
df_rosetta = pd.read_csv(join(args.data_path, args.dataset, "rosetta", args.dataset + ".tsv"), sep = "\t")
logging.info("Length of dataset: " + str(len(df_rosetta)))

#Load Amino Acid Sequence of Wildtype
with open(join(args.data_path ,'all_datasets.yml')) as file:
    all_datasets = yaml.load(file, Loader=yaml.FullLoader)
aa_wt = all_datasets[args.dataset]["wt_aa"]
logging.info("Wildtype amino acid: " + aa_wt)


columns = df_rosetta.columns
score_columns = [c for c in columns if c not in ['pdb_fn', 'mutations', 'job_uuid', 'start_time', 'run_time','mutate_run_time',
                                                  'relax_run_time', 'filter_run_time',  'centroid_run_time', "overlap_chainbreak",
                                                    "linear_chainbreak", "ss_mis", "res_count_all", 'dslf_fa13']]


n_rows = len(df_rosetta)
n_scores = len(score_columns)


mutations = df_rosetta['mutations'].values
scores = df_rosetta[score_columns].values
logging.info("Extracted scores and mutations")


# If vectorization is not possible, use numpy array operations
sequences = np.empty(n_rows, dtype=object)
for i, variant in enumerate(mutations):
    sequences[i] = generate_mutant_sequence(aa_wt, variant, offset=-1)
    if i % 10**6 == 0:
        logging.info(f"Generated {i} sequences")

scores = np.array(scores)
scores = scores.astype(np.float32)
scores = (scores - np.mean(scores, axis = 0)) / np.std(scores, axis = 0)
scores = np.delete(scores, 19, 1)
scores = np.round(scores, 3)


df_new = pd.DataFrame({"sequences": sequences, "scores": list(scores)})
logging.info("New dataframe created")

logging.info("Saving new dataframe")
df_new.to_csv(join(args.data_path, args.dataset, "rosetta", args.dataset + "_all_scores.csv"), sep= "\t", index = False)
logging.info("Preprocessing finished")