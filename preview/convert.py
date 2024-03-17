import os
import subprocess
import tqdm
import argparse


def convert_notebooks(ipynb_code_folder, output_folder):
    # Ensure the folder paths are absolute
    ipynb_code_folder = os.path.abspath(ipynb_code_folder)
    output_folder = os.path.abspath(output_folder)

    # List all .ipynb files in the source folder
    ipynb_code_files = [
        f for f in os.listdir(ipynb_code_folder) if f.endswith(".ipynb")
    ]

    # Convert each notebook to HTML
    for f in tqdm.tqdm(ipynb_code_files):
        notebook_path = os.path.join(ipynb_code_folder, f)
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                notebook_path,
                "--output-dir",
                output_folder,
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Jupyter Notebooks to HTML")
    parser.add_argument(
        "--input",
        "-i",
        help="The folder containing Jupyter Notebook files (.ipynb)",
        default="./src",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="The folder to save the converted HTML files",
        default="./preview",
    )

    args = parser.parse_args()

    convert_notebooks(args.input, args.output)
