"""Download and prepare the Landscape Pictures dataset."""

import subprocess


def main():
    """Download the dataset and extract it into the project data directory."""
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "arnaud58/landscape-pictures",
            "-p",
            "data/raw"
        ],
        check=True
    )

    print("Dataset downloaded successfully.")


if __name__ == "__main__":
    main()