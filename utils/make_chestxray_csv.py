# THIS SCRIPT SHOULD BE RUN IN THE chest_xray dataset folder

# %%
import pandas as pd
import glob
import os

# %%
def make_df(path):
    df = {}
    for file_n in glob.glob(os.path.join(path, "**/*")):
        k = file_n
        v = 0
        if 'bacteria' in k:
            v = 1
        elif 'virus' in k:
            v = 2
        df[k] = v

    return pd.DataFrame(sorted(df.items()), columns=['image_id', 'label'])

# %%
def make_balanced_train_val_df(train_df, val_df):
    df = pd.concat((train_df, val_df))

    train_df = []
    val_df = []

    for label in df.label.unique():
        if label == 0:
            train_df.append(df[df.label == label][:1000])
            val_df.append(df[df.label == label][1000:])
        else:
            train_df.append(df[df.label == label][:1000])
            val_df.append(df[df.label == label][1000:1000+349])

    return pd.concat(train_df), pd.concat(val_df)

# %%
train_df = make_df('train')
val_df = make_df('val')
test_df = make_df('test')

# %%
full_df = pd.concat((train_df, val_df))
full_df.to_csv('full.csv')

# %%
train_df, val_df = make_balanced_train_val_df(train_df, val_df)

# %%
train_df.groupby('label').count()

# %%
val_df.groupby('label').count()

# %%
test_df.groupby('label').count()

# %%
train_df.to_csv('train_3_classes.csv')
val_df.to_csv('val_3_classes.csv')
test_df.to_csv('test_3_classes.csv')
