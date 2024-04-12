import pandas as pd

from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder


class CIRI_Dataset(Dataset):

    def __init__(self, root, root_augmented=None,
                 transform=None, target_transform=None):
        base_dataset = ImageFolder(
            root=root, transform=transform, target_transform=target_transform
        )
        augmentd_dataset = ImageFolder(
            root=root_augmented, transform=transform, target_transform=target_transform
        )
        self.datasets = ConcatDataset([base_dataset, augmentd_dataset])
        self.class_to_idx = base_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = base_dataset.classes
        self.samples = base_dataset.samples + augmentd_dataset.samples
        self.targets = base_dataset.targets + augmentd_dataset.targets
        self.imgs = self.samples
        self.info, self.class_counts = self.__compute_info__()
        

    def __getitem__(self, index):
        try:
            return self.datasets[index]
        except:
            return None
    
    def __len__(self):
        return len(self.datasets)
    
    def __compute_info__(self):
        info_df = pd.DataFrame.from_records(self.samples,
                                             columns=['path', 'label'])
        info_df['label_str'] = info_df['label'].apply(lambda x: self.idx_to_class[x])
        class_counts = info_df['label_str'].value_counts()
        return info_df, class_counts
