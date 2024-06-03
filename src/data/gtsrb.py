import os
from torchvision.datasets import VisionDataset
from typing import Optional, Callable
import pandas as pd
import warnings
from PIL import Image
from dataclasses import dataclass

@dataclass
class _Item:
    filename: str = ""
    width: int = -1
    height: int = -1
    x1: int = -1
    y1: int = -1
    x2: int = -1
    y2: int = -1
    class_id: int = -1

    # implement function for sort
    def __lt__(self, other):
        return self.filename < other.filename
    
    def __gt__(self, other):
        return self.filename > other.filename

    def __eq__(self, other):
        return self.filename == other.filename

class GTSRBDatasetTrain(VisionDataset):
    """

    The German Traffic Sign Recognition Benchmark

    Optimal Image size: 32
    Number of classes: 43
    Number of channels: 3

    In detail, the annotations provide the following fields: 
 
    Filename        - Image file the following information applies to 
    Width, Height   - Dimensions of the image 
    Roi.x1,Roi.y1, 
    Roi.x2,Roi.y2   - Location of the sign within the image 
                    (Images contain a border around the actual sign 
                    of 10 percent of the sign size, at least 5 pixel) 
    ClassId         - The class of the traffic sign 

    The Train images are located in
    root/GTSRB/Training/
    - label
       - 000000_oooooo.ppm
       - ...
       - 000000_oooooo.ppm
       - GT-label.csv
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        enable_crop: Optional[bool] = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.enable_crop = enable_crop

        self._img_folder = os.path.join(root, "GTSRB", "Training")
        
        self._label_folders = []
        for _l in os.listdir(self._img_folder):
            if not os.path.isdir(os.path.join(self._img_folder, _l)):
                continue
            self._label_folders.append(_l)
        self._label_folders.sort()

        self.images = []
        for _i, _l in enumerate(self._label_folders):
            gt_csv = os.path.join(self._img_folder, _l, f"GT-{_l}.csv")
            if os.path.isfile(gt_csv):
                gt_csv = pd.read_csv(gt_csv, sep=";")
                for i, row in gt_csv.iterrows():
                    # Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                    _item = _Item(
                        os.path.join(_l, row["Filename"]),
                        row["Width"],
                        row["Height"],
                        row["Roi.X1"],
                        row["Roi.Y1"],
                        row["Roi.X2"],
                        row["Roi.Y2"],
                        row["ClassId"],
                    )
                    if os.path.isfile(os.path.join(self._img_folder, _item.filename)):
                        self.images.append(_item)
            else:
                warnings.warn(f"No csv label file found in {_l}")
                for _f in os.listdir(os.path.join(self._img_folder, _l)):
                    if _f.endswith("ppm"):
                        _item = _Item(
                            os.path.join(_l, _f),
                            class_id=int(_l)
                        )
                        self.images.append(_item)
        
        self.images.sort(key=lambda x: x.filename)

    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, index: int):
        cur_img = self.images[index]
        img = Image.open(os.path.join(self._img_folder, cur_img.filename))
        img = img.convert("RGB")
        if self.enable_crop:
            img = img.crop((cur_img.x1, cur_img.y1, cur_img.x2, cur_img.y2))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            cur_img.class_id = self.target_transform(cur_img.class_id)
        return img, cur_img.class_id

class GTSRBDatasetTest(VisionDataset):

    """

    Optimal Image size: 32
    Number of classes: 43
    Number of channels: 3

    The Test images are located in 
    root/GTSRB/Final_Test/Images

    The infomation of test images are located in 
    root/GTSRB/Final_Test/Images/GT-final_test.test.csv

    The label file is located in
    root/GT-final_test.csv

    file format:
    Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId


    Requires:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        enable_crop: We will crop the image with rigen given in the label file

    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        enable_crop: Optional[bool] = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.enable_crop = enable_crop

        self._img_folder = os.path.join(root, "GTSRB", "Final_Test", "Images")
        self._label_file = os.path.join(root, "GT-final_test.csv")

        # read the label_file
        self.labels = pd.read_csv(self._label_file, sep=";")

        # read the img_folder
        imgs_set = set(i for i in os.listdir(
            self._img_folder) if i.endswith(".ppm"))

        # find  Filename in imgs_set but not in self.labels
        none_imgs = imgs_set.difference(set(self.labels["Filename"]))
        if len(none_imgs) > 0:
            warnings.warn(
                f"There are some images in Folder {self._img_folder} is not registered in the label file {self._label_file}.\nIt is possible when some images are deleted. It's better to check and download the dataset from internet.")
            self.labels = self.labels[~self.labels["Filename"].isin(none_imgs)]

    def __getitem__(self, index: int):
        pd_row = self.labels.iloc[index]
        img_path = os.path.join(self._img_folder, pd_row["Filename"])
        img = Image.open(img_path).convert("RGB")
        if self.enable_crop:
            x1 = pd_row["Roi.X1"]
            y1 = pd_row["Roi.Y1"]
            x2 = pd_row["Roi.X2"]
            y2 = pd_row["Roi.Y2"]
            img = img.crop((x1, y1, x2, y2))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(pd_row["ClassId"])

    def __len__(self) -> int:
        return self.labels.shape[0]

def GTSRB(
        root: str,
        train: bool = True,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        enable_crop: Optional[bool] = False,
    ) -> None:
        if train:
            return GTSRBDatasetTrain(
                root=root,
                transforms=transforms,
                transform=transform,
                target_transform=target_transform,
                enable_crop=enable_crop,
            )
        else:
            return GTSRBDatasetTest(
                root=root,
                transforms=transforms,
                transform=transform,
                target_transform=target_transform,
                enable_crop=enable_crop,
            )
        
