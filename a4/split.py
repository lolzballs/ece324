import glob
import random
import functools
import shutil
import os

all_paths = glob.glob("all/*/*")
persons = set(map(lambda x: int(x.split('/')[2].split('_')[0]), all_paths))
total = len(persons)
target = int(total * 0.7)


train_persons = random.sample(persons, target)
train_paths = [path for person in train_persons for path in glob.glob(f'all/*/{person}_*')]
print(len(train_paths))

val_paths = set(all_paths).difference(train_paths)
print(len(val_paths))

for train_path in train_paths:
    target_path = f'train/{train_path.split("/", 1)[1]}'
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copyfile(train_path, target_path)

for val_path in val_paths:
    target_path = f'val/{val_path.split("/", 1)[1]}'
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copyfile(val_path, target_path)

