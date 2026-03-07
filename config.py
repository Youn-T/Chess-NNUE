version_file = open("VERSION", "r")
VERSION = version_file.read().strip()
version_file.close()

# ---------------------------------------------------------------
weights_dir = "Weights/"
formated_dataset_dir = "Dataset/Formated/"
raw_dataset_dir = "Dataset/Raw/"
log_dir = "Logs/"

labels_filename = "labels_V{version}.npz".format(version=VERSION)
moves_us_filename = "moves_us_V{version}.npz".format(version=VERSION)
moves_them_filename = "moves_them_V{version}.npz".format(version=VERSION)
weights_filename = "weights_V{version}.npz".format(version=VERSION)
raw_dataset_filename = "chessData.csv"
log_filename = "training_log_V{version}.txt".format(version=VERSION)

# ---------------------------------------------------------------
LABELS_DIR = weights_dir + labels_filename
MOVES_US_DIR = weights_dir + moves_us_filename
MOVES_THEM_DIR = weights_dir + moves_them_filename
WEIGHTS_DIR = weights_dir + weights_filename
RAW_DATASET_DIR = raw_dataset_dir + raw_dataset_filename
LOG_DIR = log_dir + log_filename