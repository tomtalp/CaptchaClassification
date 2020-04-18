import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io

def get_metadata_df(images_path):
    """
    Returns a Pandas DF containing all the metadata for our PyTorch dataset.
    This DF has 2 columns - An image path and it's label. 
    The label can be one of 62 characters - 26 English letters * 2 (upper/lowercase) + 10 digits = 62

    @param images_path (str) - The path from which image paths will be collected
    @return Pandas DF (<img_path:str>, <label:str>)
    """
    dataset_images = glob.glob("{base_dataset}/*.png".format(base_dataset=images_path))

    images_data_for_df = []
    for img in dataset_images:
        # The path we're getting is the full path - /path/to/img.png. Split by `/` and get the last part - that's our image name!
        filename = img.split("/")[-1] 

        # Our file names are of the following format - <char>_<random_id>.png
        # Extract the char name by splitting via the `_` char
        label = filename.split("_")[0]

        info = {
            "img_path": img,
            "label": label
        }
        images_data_for_df.append(info)

    df = pd.DataFrame(images_data_for_df)
    return df

class CaptchaSingleLettersDataset(Dataset):
    """
    Single letter images dataset, based on the PyTorch Dataset object
    """
    def __init__(self, dataset_metadata_df):
        """
        @param dataset_metadata_df (Pandas DF) - A DF representing all our metadata for this Dataset - image paths & labels
        """
        self.dataset_metadata_df = dataset_metadata_df

    def __len__(self):
        return len(self.dataset_metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_metadata = self.dataset_metadata_df.iloc[idx]
        img_path = img_metadata[0]
        label = img_metadata[1]
        image = io.imread(img_path) # This returns an ndarray of size WxHxC (3 channels here, since it's RGB)
        return {"image": image, "label": label}
        



if __name__ == "__main__":
    train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/train_dataset/'
    test_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/test_dataset/'

    train_dataset_metadata_df = get_metadata_df(train_dataset_path)
    train_dataset = CaptchaSingleLettersDataset(train_dataset_metadata_df)


    


