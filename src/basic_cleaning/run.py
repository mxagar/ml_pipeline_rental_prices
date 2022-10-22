#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, 
exporting the result to a new artifact.
"""
import os
import argparse
import logging

import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    project_name = "nyc_airbnb"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.environ["WANDB_PROJECT"]
    run = wandb.init(project=project_name, job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading artifact: %s", args.input_artifact)
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    logger.info("Cleaning artifact: %s", args.input_artifact)    
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # Temporary file
    filename = "clean_sample.csv"
    df.to_csv(filename, header=True, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact: %s", args.output_artifact)
    run.log_artifact(artifact)

    # Remove created temporary file
    # we could also use tempfile.NamedTemporaryFile()
    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Filename of the input artifact (the raw dataset)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Filename of the output artifact (cleaned dataset)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price in USD allowed in the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price in USD allowed in the dataset",
        required=True
    )


    args = parser.parse_args()

    go(args)