from datasets import load_dataset

dataset = "tomekkorbak/python-github-code"

ds = load_dataset(dataset, split='train')

id_column = []

for i in range(len(ds)):
    id_column.append(i)

lis = ds.column_names
lis.remove("text")

ds = ds.add_column("id", id_column).remove_columns(lis)

ds = ds.select_columns(["id"] + [name for name in ds.column_names if name != "id"])

ds = ds.select(range(1000))
ds.to_csv(dataset.split("/")[1]+"-data.csv")

