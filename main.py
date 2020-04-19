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
    
class P_M_CaptchaDataset(Dataset):
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
        if char_label.lower() == 'm':
            char_label_idx = 0
        else:
            char_label_idx = 1
        # char_label_idx = self.list_of_possible_chars.index(char_label) # Get the index of that character, from the possible chars array
        # label_as_tensor = torch.zeros([len(self.list_of_possible_chars)]) # Initialize a tensor of zeros, the size of possible chars
        # label_as_tensor[char_label_idx] = 1 # Set the tensor index of the char label to 1, and everything else is 0

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


class SingleLetterCaptchaDeepCNN(nn.Module):
    def __init__(self):
        super(SingleLetterCaptchaDeepCNN, self).__init__()

        self.conv1_layer = nn.Conv2d(3, 32, 5) 
        self.conv2_layer = nn.Conv2d(32, 16, 3) 
        self.conv3_layer = nn.Conv2d(16, 16, 3) 
        self.conv4_layer = nn.Conv2d(16, 8, 2) 
        # self.conv5_layer = nn.Conv2d(526, 526, 2) 

        self.layer_size_after_convs = 8 * 1 * 1 # hardcoded for now, implement in `get_layer_size_after_convs` later
        self.fc1 = nn.Linear(self.layer_size_after_convs, TOTAL_NUM_OF_CLASSES)
        # self.fc2 = nn.Linear(124, TOTAL_NUM_OF_CLASSES)
        
        # self.final_fc_layer = nn.Linear(self.layer_size_after_convs , TOTAL_NUM_OF_CLASSES)
    
    def forward(self, x):
        """
        Perform a forward pass on the network
        """
        x = F.relu(self.conv1_layer(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv2_layer(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv3_layer(x))
        x = F.max_pool2d(x, (2, 2))
    
        x = F.relu(self.conv4_layer(x))
        x = F.max_pool2d(x, (3, 3))

        # x = F.relu(self.conv5_layer(x))
        # x = F.max_pool2d(x, (2, 2))

        # Reshape tensor so that it's still in batches, but each conv result is now one big vector
        x = x.view(-1, self.layer_size_after_convs)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.sofmtax(x, dim=1) # Softmax along dimension 1 (that is, the values of every single object in the batch, and not across batches)
        x = torch.sigmoid(x)

        return x



conv1_layer = nn.Conv2d(3, 32, 5) # 3 input channels, 64 output channels, 3x3 kernel
conv2_layer = nn.Conv2d(32, 16, 3) # 64 input channels, 128 output, 3x3 kernel
conv3_layer = nn.Conv2d(16, 16, 3) # 128 input channels, 256 output, 3x3 kernel
conv4_layer = nn.Conv2d(16, 8, 2) # 256 input channels, 512 output, 3x3 kernel
# conv5_layer = nn.Conv2d(512, 512, 2) # 512 input channels, 512 output, 3x3 kernel

x = img_batch
x = F.relu(conv1_layer(x))
x = F.max_pool2d(x, (3, 3))

x = F.relu(conv2_layer(x))
x = F.max_pool2d(x, (2, 2))

x = F.relu(conv3_layer(x))
x = F.max_pool2d(x, (2, 2))

x = F.relu(conv4_layer(x))
x = F.max_pool2d(x, (3, 3))

# x = F.relu(conv5_layer(x))
# x = F.max_pool2d(x, (2, 2))

def test_single_letters():
    TOTAL_NUM_OF_CLASSES = 2
    train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/train_dataset/'
    train_dataset_metadata_df = get_metadata_df(train_dataset_path)
    train_dataset_metadata_df = train_dataset_metadata_df[train_dataset_metadata_df['label'].isin(['m', 'M', 'p', 'P'])]
    train_dataset_metadata_df = train_dataset_metadata_df.reset_index(drop=True)
    train_dataset_metadata_df = train_dataset_metadata_df.head(2)
    train_dataset = P_M_CaptchaDataset(train_dataset_metadata_df)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    test_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/test_dataset/'
    test_dataset_metadata_df = get_metadata_df(test_dataset_path)
    test_dataset_metadata_df = test_dataset_metadata_df[test_dataset_metadata_df['label'].isin(['m', 'M', 'p', 'P'])]
    test_dataset_metadata_df = test_dataset_metadata_df.reset_index(drop=True)
    test_dataset = P_M_CaptchaDataset(test_dataset_metadata_df)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = SingleLetterCaptchaDeepCNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, weight_decay=0.1)

    total_steps = len(train_dataset_loader)

    epochs = 50
    for epoch_num in range(epochs):
        for i, (img_batch, labels) in enumerate(train_dataset_loader):
            fw_pass_output = model(img_batch)
            loss_values = loss_func(fw_pass_output, labels)

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_num+1, epochs, i+1, total_steps, loss_values.item()))
    
    model.eval()
    total_test_samples = len(test_dataset)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataset_loader:
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = (correct / total) * 100
        print('Test Accuracy of the model on the {count} test images: {acc}'.format(count=total_test_samples, acc = accuracy))



if __name__ == "__main__":
    train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/train_dataset/'
    test_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/test_dataset/'

    train_dataset_metadata_df = get_metadata_df(train_dataset_path)
    train_dataset = CaptchaSingleLettersDataset(train_dataset_metadata_df)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    test_dataset_metadata_df = get_metadata_df(test_dataset_path)
    test_dataset = CaptchaSingleLettersDataset(test_dataset_metadata_df)

    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

    # model = SingleLetterCaptchaCNN()
    model = SingleLetterCaptchaDeepCNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, weight_decay=0.1)

    total_steps = len(train_dataset_loader)

    epochs = 10
    for epoch_num in range(epochs):
        for i, (img_batch, labels) in enumerate(train_dataset_loader):
            fw_pass_output = model.forward(img_batch)
            loss_values = loss_func(fw_pass_output, labels)

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_num+1, epochs, i+1, total_steps, loss_values.item()))


# model.eval()
# total_test_samples = len(test_dataset)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_dataset_loader:
#         outputs = model(images.float())
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     accuracy = (correct / total) * 100
#     print('Test Accuracy of the model on the {count} test images: {acc}'.format(count=total_test_samples, acc = accuracy))


    

# conv1_layer = nn.Conv2d(3, 10, 5) # 3 input channels, 10 output channels, 5x5 kernel
# conv2_layer = nn.Conv2d(10, 2, 3) # 10 input channels, 2 output, 3x3 kernel

# x = F.relu(conv1_layer(images))
# x = F.max_pool2d(x, (2, 2))

# x = F.relu(conv2_layer(x))
# x = F.max_pool2d(x, (4, 4))