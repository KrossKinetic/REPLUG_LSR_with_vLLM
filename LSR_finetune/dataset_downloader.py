import csv
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split='train')

# Define the output TSV file name
output_filename = "psgs_w100.tsv"

# Get the field names from the dataset features
# This makes the code adaptable to other datasets
fieldnames = list(dataset.features.keys())

# Write the data to a TSV file
with open(output_filename, 'w', newline='', encoding='utf-8') as tsvfile:
    # Create a TSV writer object
    writer = csv.writer(tsvfile, delimiter='\t')

    # Write the header row
    writer.writerow(fieldnames)

    # Iterate through the dataset with a progress bar and write each row
    for row in tqdm(dataset, desc="Converting to TSV"):
        # The dataset returns a dictionary for each row.
        # We extract the values in the correct order.
        writer.writerow([row[field] for field in fieldnames])

print(f"\nSuccessfully converted the dataset to {output_filename}")