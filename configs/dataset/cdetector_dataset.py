# dataset settings 
# data_root = 'cervix/ComparisonDetectorDataset'
# data_root = 'CervicalDatasets/ComparisonDetectorDataset'
data_root = 'CervicalDatasets/ComparisonDetectorDataset'
classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
num_classes = len(classes)
train_bs = 128
val_bs = 128


train_annojson = f'{data_root}/annofiles/train_patches.json'
val_annojson = f'{data_root}/annofiles/val_annojson.json'