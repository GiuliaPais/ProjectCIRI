#import datautils
import os
# import torch
# import pandas as pd
# import torchvision

# from ray.train.torch import prepare_data_loader

from ciri_utils import datautils, engine_v2


root_base = os.path.abspath('/Users/giulia/Library/CloudStorage/GoogleDrive-giuliapais1@gmail.com/My Drive/ProjectCIRI/Incidents-subset')
root_augmented = os.path.abspath('/Users/giulia/Library/CloudStorage/GoogleDrive-giuliapais1@gmail.com/My Drive/ProjectCIRI/augmented_images')

# ciri_dataset = datautils.CIRI_Dataset(
#     root=root_base, root_augmented=root_augmented)
# print(ciri_dataset.info)
# sub = torch.utils.data.Subset(ciri_dataset, range(100))
# # split = torch.utils.data.random_split(sub, [0.8, 0.2])
# # print(len(split[0]))
# print(sub.dataset)
# print(engine_v2._find_classes(sub))

# df = pd.DataFrame(columns=['x', 'y', 'z'])
# #df = df.append({'x': 1, 'y': 2, 'z': 3}, ignore_index=True)
# row = pd.Series({'x': 1, 'y': 2, 'z': 3})
# df = pd.concat([df, row.to_frame().T], ignore_index=True)
# print(df)
# model = torchvision.models.get_model('resnet18', num_classes=10)
# print(model.parameters())

idx, ds = engine_v2._sample_indices_from_prop(0.1, root_base=root_base, root_augmented=root_augmented)
# print(len(idx))
# print(len(ds))
# print(idx)

# gen = engine_v2._get_fold_indices(ds.info, sample_idx=idx, outer_cv_k=5, inner_cv_k=5)
# print(gen)
# print(next(gen))

gen = engine_v2._get_fold_indices(ds.info, sample_idx=list(range(1000)), outer_cv_k=3)
for i, j in gen:
    print(f"All in range i: {all([x in range(1000) for x in i])}")
    print(f"All in range j: {all([x in range(1000) for x in j])}")

# from sklearn.model_selection import StratifiedKFold
# kf = StratifiedKFold(5, shuffle=True, random_state=42)
# print(ds.info)
# gen = kf.split(ds.info.index, ds.info['label_str'])
# print(next(gen))