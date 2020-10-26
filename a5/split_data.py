"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

import pandas as pd
import numpy as np

SEED = 0

df = pd.read_csv('./data/data.tsv', sep='\t')
objective = df.loc[df['label'] == 0]
subjective = df.loc[df['label'] == 1]

target_overfit = 50 // 2
target_train = int(len(df) * 0.32)
target_val = int(len(df) * 0.10)
target_test = int(len(df) * 0.08)

overfit_set = pd.concat([
        objective.sample(n=target_overfit, random_state=SEED),
        subjective.sample(n=target_overfit, random_state=SEED)
])
print('overfit_set label counts')
print(overfit_set['label'].value_counts())


print(f'Splitting {len(df)} samples into {target_train*2} training, {target_val*2} validation, {target_test*2} testing')

obj_training = objective.sample(n=target_train, random_state=SEED)
sub_training = subjective.sample(n=target_train, random_state=SEED)
objective = objective.drop(obj_training.index)
subjective = subjective.drop(sub_training.index)
train_set = pd.concat([obj_training, sub_training])
print('train_set label counts')
print(train_set['label'].value_counts())

obj_val = objective.sample(n=target_val, random_state=SEED)
sub_val = subjective.sample(n=target_val, random_state=SEED)
objective = objective.drop(obj_val.index)
subjective = subjective.drop(sub_val.index)
val_set = pd.concat([obj_val, sub_val])
print('val_set label counts')
print(val_set['label'].value_counts())

obj_test = objective.sample(n=target_test, random_state=SEED)
sub_test = subjective.sample(n=target_test, random_state=SEED)
objective = objective.drop(obj_test.index)
subjective = subjective.drop(sub_test.index)
test_set = pd.concat([obj_test, sub_test])
print('test_set')
print(test_set['label'].value_counts())

print(f'{len(objective)} objective and {len(subjective)} subjective samples left over')

train_set.to_csv('./data/train.tsv', sep='\t', index=False)
val_set.to_csv('./data/validation.tsv', sep='\t', index=False)
test_set.to_csv('./data/test.tsv', sep='\t', index=False)
overfit_set.to_csv('./data/overfit.tsv', sep='\t', index=False)


