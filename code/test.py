import datautils
import os

root_base = os.path.abspath('./Incidents-subset')
root_augmented = os.path.abspath('./augmented_images')

ciri_dataset = datautils.CIRI_Dataset(root=root_base, root_augmented=root_augmented)
# print(ciri_dataset.classes)
# print(ciri_dataset.class_to_idx)
# print(ciri_dataset[0])
# print(ciri_dataset.samples[1])
# print(ciri_dataset.targets[1])
print(ciri_dataset.info.head())
print(ciri_dataset.class_counts)