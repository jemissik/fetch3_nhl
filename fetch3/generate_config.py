from __future__ import annotations
from pathlib import Path

from attrs import fields
import click
import pandas as pd
from fetch3.model_config import SCHEMES
import yaml

# All currently used parameter classes
PARAMETER_CLASSES = [value["parameters"] for value in SCHEMES.values()]



def generate_config_from_csv(csv_path: Path, output_path: Path, position: list[int], **kwargs):
    cls_fields = set(attr.name for cls in PARAMETER_CLASSES for attr in fields(cls))
    cls_fields = cls_fields.union({"model_tree", "model_trees"})
    df = pd.read_csv(csv_path, **kwargs)
    df = df.iloc[position]

    df_cols = df.columns
    df_cols = [col for col in df_cols if col in cls_fields]
    df = df[df_cols]


    col = None
    if "model_tree" in df.columns:
        col = "model_tree"
    elif "model_trees" in df.columns:
        col = "model_tree"

    if col:
        df["cumcount"] = df.groupby(col).cumcount() + 1
        df[col] = df[col] + df["cumcount"].astype(str)
        df = df.set_index(col)
        df = df.drop(columns=["cumcount"])
    else:
        df = df.reset_index(drop=True)
    config = """
model_options:
   ...

model_trees:"""

    for tree_name, row in df.iterrows():
        config += f"""
    {tree_name}:"""

        for param_name, param_value in row.items():
            config += f"""
        {param_name}: {param_value}"""

    with open(output_path, "w") as f:
        # Write model options from loaded config
        # Parameters for the trial from Ax
        yaml.dump(yaml.safe_load(config), f)
        return f.name


@click.command()
@click.option(
    "-cp",
    "--csv-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to csv file",
)
@click.option(
    "-op",
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default="config.yml",
    help="Path to output config file",
)
@click.option(
    "-p",
    '--position',
    multiple=True,
    type=int,
    help="Position of the tree is in config file (Row the csv file)"
         "(0 based indexing, first row is 0)."
         "Can assign multiple if you have multiple trees in the csv file (ex: -p 4 -p 6 -p 17),",
)
@click.option(
    "-d",
    '--delimiter',
    type=str,
    help=r"Delimiter of the csv file (Default is ',', "
         r"can be changed to '\t' for tab-delimited files or '\s+' for space-delimited files for example)",
)
@click.option(
    "-hr",
    '--header-row',
    type=int,
    # multiple=True,
    help="Row number of the header (Default is to infer header row).",
)
def main(csv_path, output_path, position, delimiter, header_row):
    kwargs = {}
    position = list(position)
    if delimiter:
        kwargs['delimiter'] = delimiter
    if header_row:
        kwargs['header'] = header_row

    position = [pos - ((header_row if header_row else 0) + 1) for pos in position]

    generate_config_from_csv(csv_path, output_path, position, **kwargs)


if __name__ == "__main__":
    main()
