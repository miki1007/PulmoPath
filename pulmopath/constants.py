"""Project-wide class mapping and preprocessing defaults."""

CLASS_TO_ID = {"lung_n": 0, "lung_aca": 1, "lung_scc": 2}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
CLASS_NAMES = [ID_TO_CLASS[i] for i in range(len(ID_TO_CLASS))]
IMAGE_SIZE = 128
RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

GFG_ARTICLE_URL = (
    "https://www.geeksforgeeks.org/deep-learning/"
    "lung-cancer-detection-using-convolutional-neural-network-cnn/"
)
GFG_DATASET_URL = (
    "https://drive.google.com/drive/folders/"
    "1vcw8JRQr8-62IVvix4rK4rDXrFnlap8e?usp=sharing"
)
