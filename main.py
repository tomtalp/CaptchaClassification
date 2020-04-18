import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
import string

TOTAL_NUM_OF_CLASSES = 62 # 62 possible Captcha characters - 26 English letters * 2 (upper/lowercase) + 10 digits = 62

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
    def __init__(self, dataset_metadata_df, img_size=80):
        """
        @param dataset_metadata_df (Pandas DF) - A DF representing all our metadata for this Dataset - image paths & labels
        @param img_size (int, default 80) = The size of an input image, assuming it's a square
        """
        self.dataset_metadata_df = dataset_metadata_df
        self.img_size = img_size
        self.list_of_possible_chars = list(string.ascii_letters + string.digits)

    def __len__(self):
        return len(self.dataset_metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_metadata = self.dataset_metadata_df.iloc[idx]
        img_path = img_metadata[0]

        char_label = img_metadata[1] # Label from pandas df is a character, for example `b`, `Z`, `3` and so on.
        char_label_idx = self.list_of_possible_chars.index(char_label) # Get the index of that character, from the possible chars array
        label_as_tensor = torch.zeros([len(self.list_of_possible_chars)]) # Initialize a tensor of zeros, the size of possible chars
        label_as_tensor[char_label_idx] = 1 # Set the tensor index of the char label to 1, and everything else is 0

        image = io.imread(img_path) # This returns an ndarray of size WxHxC (3 channels here, since it's RGB)
 
        image = torch.from_numpy(image) # Convert image to a Torch tensor object
        image = image.reshape([3, self.img_size, self.img_size]) # Convert to CxWxH (That's what torch assumes)
        image = image.float() # Convert to type float
        
        # return (image, label_as_tensor.long())

        return (image, char_label_idx)
    

class SingleLetterCaptchaCNN(nn.Module):
    def __init__(self):
        super(SingleLetterCaptchaCNN, self).__init__()

        self.conv1_layer = nn.Conv2d(3, 10, 5) # 3 input channels, 10 output channels, 5x5 kernel
        self.conv2_layer = nn.Conv2d(10, 2, 3) # 10 input channels, 2 output, 3x3 kernel

        self.layer_size_after_convs = 2 * 9 * 9 # hardcoded for now, implement in `get_layer_size_after_convs` later
        self.final_fc_layer = nn.Linear(self.layer_size_after_convs , TOTAL_NUM_OF_CLASSES)
    
    def forward(self, x):
        """
        Perform a forward pass on the network
        """
        x = F.relu(self.conv1_layer(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv2_layer(x))
        x = F.max_pool2d(x, (4, 4))

        # Reshape tensor so that it's still in batches, but each conv result is now one big vector
        x = x.view(-1, self.layer_size_after_convs)
        x = self.final_fc_layer(x)
        x = F.softmax(x, dim=1) # Softmax along dimension 1 (that is, the values of every single object in the batch, and not across batches)

        return x


    # def get_layer_size_after_convs(self):
    #     """
    #     Get the size of the final layer after all convolutions.
    #     TODO - THIS IS MANUALLY CALCULATED RIGHT NOW, DO THIS DYNAMICALLY
    #     """
    #     # Initial input - 3x80x80. 3 channels (RGB), 80x80 image
    #     initial_input_size = 3 * 80 * 80

    #     size_after_conv1 = 10 * 76 * 76
    #     size_after_conv1_maxpool = 10 * 38 * 38





if __name__ == "__main__":
    train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/train_dataset/'
    test_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/test_dataset/'

    train_dataset_metadata_df = get_metadata_df(train_dataset_path)
    train_dataset = CaptchaSingleLettersDataset(train_dataset_metadata_df)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    model = SingleLetterCaptchaCNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    total_steps = len(train_dataset_loader)

    epochs = 10
    for epoch_num in range(epochs):
        for i, (img_batch, labels) in enumerate(train_dataset_loader):
            fw_pass_output = model.forward(img_batch)
            loss_values = loss_func(fw_pass_output, labels)

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_num+1, epochs, i+1, total_step, loss_values.item()))



    

# conv1_layer = nn.Conv2d(3, 10, 5) # 3 input channels, 10 output channels, 5x5 kernel
# conv2_layer = nn.Conv2d(10, 2, 3) # 10 input channels, 2 output, 3x3 kernel

# x = F.relu(conv1_layer(images))
# x = F.max_pool2d(x, (2, 2))

# x = F.relu(conv2_layer(x))
# x = F.max_pool2d(x, (4, 4))