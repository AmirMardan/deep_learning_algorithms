import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, random_split, DataLoader
import os 
import numpy as np
from typing import Tuple, Any, Union
from PIL import Image
import torch
from torch import Tensor

def _get_loaders(dataset: Dataset,
                batch_size: int,
                val_percentage: float = 0.1,
                seed: int=10):
    """
    _get_loaders gets the dataset and split it based on `val_percentage`
    and returns training and validation loaders

    Parameters
    ----------
    dataset : Dataset
        _description_
    batch_size : int
        _description_
    val_percentage : float, optional
        _description_, by default 0.1
    seed : int, optional
        _description_, by default 10

    Returns
    -------
    train_loader : DataLoader
        _description_
        
    val_loader : DataLoader
    """
        
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
    """
    test_preparing for testing the final results 

    _extended_summary_

    Parameters
    ----------
    img_path : str
        _description_
    img_height : Any
        _description_
    img_width : Any
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    transform = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
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
               ) -> Union[DataLoader, DataLoader]:
    """
    get_loaders gets the base directory of the data and provides
    training and validation loaders

    _extended_summary_

    Parameters
    ----------
    base_dir : str
        _description_
    img_height : Any
        _description_
    img_width : Any
        _description_
    batch_size : int
        _description_
    val_percentage : float
        _description_
    p : float
        _description_

    Returns
    -------
    train_loader : DataLoader
        _description_
        
    val_loader : DataLoader
    """
    
    data_dir = [base_dir +"/data" + i + "/data" + i for i in ["A", "B", "C", "D", "E"]]
    
    imgs_path = [i + "/CameraRGB/" for i in data_dir]
    masks_path = [i + "/CameraSeg/" for i in data_dir]
    
    
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
    """
    get_transform prepares transformer for data

    Parameters
    ----------
    img_height : Any
        _description_
    img_width : Any
        _description_
    p : float
        _description_

    Returns
    -------
    _type_
        _description_
    """
    train_transform = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Rotate(limit=35, p=p),
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
    """
    CarvanaDataset is a custom dataset for this problem

    Parameters
    ----------
    img_dir : str
        _description_
    mask_dir : str
        _description_
    transform : A.Compose, optional
        _description_, by default None
    """
    def __init__(self, 
             img_dir: str,
             mask_dir: str,
             transform: A.Compose = None) -> None:
        
        super().__init__()

        
        self.imgs_name = []
        self.masks_name = []
        for i in range(len(img_dir)):
            imgs = os.listdir(img_dir[i])
        
            self.imgs_name.extend(
                [img_dir[i] + img for img in imgs if img!=".DS_Store"]
                )
            self.masks_name.extend(
                [mask_dir[i] + img for img in imgs if img!=".DS_Store"]
                )
        
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs_name) 
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        img = np.array(Image.open(self.imgs_name[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks_name[index]).convert("L"), np.float32)
        
        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        
        return img, mask
    
    
def test():
    """
    test is a function to test this module

    """
    import matplotlib.pyplot as plt
    
    BATCH_SIZE = 1
    base_dir = os.path.abspath(os.path.join(__file__, "../../../data/lyft_udacity"))
    
    train_loader, val_loader = get_loaders(
        base_dir, img_height=572, img_width=572,
        batch_size=BATCH_SIZE, val_percentage=0.1, p=0.0
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
    
    
    
    
    