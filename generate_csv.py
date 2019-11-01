import csv
from pathlib import Path

root_dir = Path('/home/danushkumar/Downloads/dataset/dog_cats/sample/')
train_valid_path = ['train', 'valid']


with open('train_data.csv', 'w') as train_csv:
    writer = csv.writer(train_csv, dialect='excel')
    writer.writerow(['img_path', 'y'])

    for p in train_valid_path: 
        path = root_dir.joinpath(p)
        
        for category in path.iterdir():
            for image_path in path.joinpath(category).glob('*.jpg'): 
                writer.writerow([image_path, category.name])
