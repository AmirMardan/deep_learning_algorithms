import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, random_split, DataLoader
import os 
import numpy as np
from numpy import ndarray
from typing import Tuple, Any
from PIL import Image
import torch

def _get_loaders(dataset: Dataset,
                batch_size: int,
                val_percentage: float = 0.1,
                seed: int=10):
        
    train_data, val_data = random_split(
        dataset=dataset,
        lengths=[1-val_percentage, val_percentage],
        generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(
        train_data, batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size, shuffle=False
        )
    
    return train_loader, val_loader


def test_preparing(img_path: str,
                   img_height: Any,
                    img_width: Any
                    ):
    
    transform = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    img = np.array(Image.open(img_path).convert("RGB"))
    return transform(image=img)["image"].unsqueeze_(0)
    
    
def get_loaders(base_dir: str,
               img_height: Any,
               img_width: Any,
               batch_size: int,
               val_percentage: float,
               p: float,
               ):
    
    imgs_path = base_dir + "/imgs"
    masks_path = base_dir + "/masks"
    transform = get_transform(img_height=img_height,
                              img_width=img_width,
                              p=p)
    
    training_data = CarvanaDataset(imgs_path,
                                   masks_path,
                                   transform=transform)
    
    train_loader, val_loader = _get_loaders(training_data,
                                           val_percentage=val_percentage,
                                           batch_size=batch_size
                                           )
    return train_loader, val_loader


def get_transform(img_height: Any,
                  img_width:Any,
                  p:float):
    train_transform = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Rotate(limit=35, p=1.0),
        A.VerticalFlip(p=p),
        A.HorizontalFlip(p=p),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    return train_transform
    
    
class CarvanaDataset(Dataset):
    def __init__(self, 
             img_dir: str,
             mask_dir: str,
             transform: A.Compose = None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_name = os.listdir(img_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_name) 
    
    def __getitem__(self, index) -> Tuple[ndarray, ndarray]:
        img_path = os.path.join(self.img_dir, self.img_name[index])
        mask_path = os.path.join(self.mask_dir, self.img_name[index].replace(".jpg", "_mask.gif"))
        
        img = np.array(Image.open(img_path).convert("RGB"))
        # img = np.transpose(img, axes=(2,0,1))
        mask = np.array(Image.open(mask_path).convert("L"), np.float32)
        mask[mask == 255.] = 1.0
        
        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        
        return img, mask
    
    
def test():
    import matplotlib.pyplot as plt
    
    BATCH_SIZE = 1
    img_dir = os.path.abspath(os.path.join(__file__, "../../../data/carvana"))
    imgs_path = img_dir + "/imgs/"
    masks_path = img_dir + "/masks/"
    
    train_loader, val_loader = get_loaders(
        imgs_path, masks_path, img_height=572, img_width=572,
        batch_size=BATCH_SIZE, val_percentage=0.1, p=0.5
    )
    
    print(f"Train data: {len(train_loader)}\nValidation data: {len(val_loader)}")
    # Display image and label.
    img, mask = next(iter(train_loader))
    
    print(img.shape, mask.shape)
        
    fig, axs = plt.subplots(1, 2, figsize=(4, 2))
    count = 0
    axs[0].imshow(img[count, ...].detach().permute(1, 2, 0))
    axs[1].imshow(mask[count, ...].detach())
    [axs[i].axis('off') for i in range(2)]
    
    axs[0].set_title("Image")
    axs[1].set_title("Mask")
        
    print(f"Max: {img.max()}, {mask.max()}")
    print(f"Min: {img.min()}, {mask.min()}")
    plt.show()
    
    
if __name__ == "__main__":
    test()
    
    
    
    
    