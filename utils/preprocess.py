from .dataset import CVCDataset, KVASDataset, ISICDataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),transforms.ToTensor(),
    ])

# Initialize datasets
# cv_dataset = CVCDataset(cv_image_folder, cv_mask_folder,transform=data_transforms)
# kv_dataset = KVASDataset(kv_image_folder, kv_mask_folder,transform=data_transforms)
dataset = ISICDataset(image_folder, mask_folder,transform=data_transforms)

# Split the dataset into train, validation, and test sets
def split_data(dataset):
    
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, 
                                                        [train_size, val_size, test_size])
    
    # Create data loaders
    bs = 16
    n_w=4
    # Adjust batch size for multi-GPU
    bs = bs * torch.cuda.device_count()
    
    # Create data loaders with multi-processing support
    train_loader = DataLoader(train_dataset, batch_size=bs, 
                   shuffle=True,num_workers=n_w,pin_memory=True,
                   drop_last=True  # Ensures consistent batch sizes across GPUs
                   )
    
    val_loader = DataLoader(val_dataset, batch_size=bs, 
                 shuffle=False,num_workers=n_w,pin_memory=True,
                 drop_last=True)
    
    test_loader = DataLoader(test_dataset, batch_size=bs, 
                  shuffle=False,num_workers=n_w,pin_memory=True,
                  drop_last=True)
    
    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")
    print(f"Total test images: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
  
