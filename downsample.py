import csv
import random
import argparse
from tqdm import tqdm

def downsample_tsv(input_path, output_path, keep_percentage=None, num_passages=None):
    """
    Reads a large TSV file and writes a smaller, randomly sampled version to a new file.

    This function is memory-efficient as it processes the file line by line.

    Args:
        input_path (str): The path to the large input .tsv file.
        output_path (str): The path where the smaller output .tsv file will be saved.
        keep_percentage (float, optional): The fraction of rows to keep (e.g., 0.3 for 30%).
        num_passages (int, optional): The exact number of passages to keep.
    """
    if keep_percentage is None and num_passages is None:
        raise ValueError("You must specify either --keep (percentage) or --num_passages (fixed number).")
        
    if keep_percentage is not None and num_passages is not None:
        raise ValueError("You can only use one of --keep or --num_passages, not both.")

    print(f"Reading from '{input_path}'...")

    try:
        # First, count the total number of lines for progress bar and sampling
        with open(input_path, 'r', encoding='utf-8') as f:
            # -1 to exclude the header from the data line count
            total_lines = sum(1 for line in f) - 1

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            # 1. Read and write the header line unconditionally
            header = next(reader)
            writer.writerow(header)

            if num_passages:
                print(f"Sampling exactly {num_passages} passages from {total_lines} total.")
                if num_passages > total_lines:
                    raise ValueError(f"--num_passages ({num_passages}) cannot be greater than the total number of lines in the file ({total_lines}).")
                
                # Generate a set of unique random line indices to keep
                indices_to_keep = set(random.sample(range(total_lines), k=num_passages))

                # Iterate with an index and keep only the selected lines
                for i, row in enumerate(tqdm(reader, total=total_lines, desc="Sampling by index")):
                    if i in indices_to_keep:
                        writer.writerow(row)
            else: # Use percentage-based sampling
                print(f"Will keep approximately {keep_percentage:.0%} of the data.")
                lines_written = 0
                for row in tqdm(reader, total=total_lines, desc="Downsampling by percentage"):
                    if random.random() < keep_percentage:
                        writer.writerow(row)
                        lines_written += 1

        print("\nDownsampling complete.")
        print(f"New corpus saved to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: The input file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A memory-efficient script to downsample a large TSV file by percentage or a fixed number."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="psgs_w100_small.tsv",
        help="Path to the large input .tsv file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="psgs_w100_small2.tsv",
        help="Path to save the smaller output .tsv file."
    )
    parser.add_argument(
        "--keep",
        type=float,
        default=None,
        help="Fraction of the data to keep (e.g., 0.3 to keep 30%%). Mutually exclusive with --num_passages."
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=None,
        help="Exact number of passages to keep. Mutually exclusive with --keep."
    )

    args = parser.parse_args()

    downsample_tsv(args.input_file, args.output_file, args.keep, args.num_passages)
