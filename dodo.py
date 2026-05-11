"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Project: Replicating Chen, Kelly, and Xiu (2022): Expected Returns and Large Language Models
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################

## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import glob
import shutil

from os import environ
from pathlib import Path

from settings import config
from pull_market_data import USE_CRSP

DOIT_CONFIG = {
    "backend": "sqlite3",
    "dep_file": "./.doit-db.sqlite",
}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
# USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace '{notebook_path}'"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir='{output_dir}' '{notebook_path}'"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir='{output_dir}' '{notebook_path}'"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace '{notebook_path}'"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv '{from_path}' '{to_path}'"
    else:
        command = f"move '{from_path}' '{to_path}'"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }

def task_pull_fred():
    """Pull data from external sources"""
    yield {
        "name": "fred",
        "doc": "Pull data from FRED",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_fred.py",
        ],
        "targets": [DATA_DIR / "fred.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_fred.py"],
        "clean": [],
    }

def task_pull_crsp():
    yield {
        "name": "crsp",
        "doc": "Pull CRSP stock and index data from WRDS",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_CRSP_stock.py",
        ],
        "targets": [
            DATA_DIR / "CRSP_MSF_INDEX_INPUTS.parquet",
            DATA_DIR / "CRSP_MSIX.parquet",
            DATA_DIR / "CRSP_DSI.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_stock.py"],
        "clean": [],
    }
    yield {
        "name": "crsp_daily",
        "doc": "Pull CRSP daily stock returns from WRDS",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_CRSP_daily.py",
        ],
        "targets": [DATA_DIR / "CRSP_daily_stock.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_daily.py"],
        "clean": [],
    }
    market_data_file_dep = [
        "./src/settings.py",
        "./src/pull_market_data.py",
    ]
    if USE_CRSP:
        market_data_file_dep.append(DATA_DIR / "CRSP_DSI.parquet")
    yield {
        "name": "market_data",
        "doc": "Compute monthly market return and volatility",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_market_data.py",
        ],
        "targets": [DATA_DIR / "market_data.parquet"],
        "file_dep": market_data_file_dep,
        "clean": [],
    }

def task_pull_sp500_constituents():
    yield {
        "name": "sp500_constituents",
        "doc": "Pull historical S&P 500 constituents from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_sp500_constituents.py",
        ],
        "targets": [
            DATA_DIR / "sp500_constituents.parquet",
            DATA_DIR / "sp500_names_lookup.parquet",
        ],
        "file_dep": [
            "./src/settings.py", 
            "./src/pull_sp500_constituents.py"
        ],
        "clean": [],
        "verbosity": 2,
    }

def task_pull_ravenpack():
    yield {
        "name": "ravenpack",
        "doc": "Pull RavenPack headlines from WRDS",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_ravenpack.py",
        ],
        "targets": [
            DATA_DIR / "ravenpack_djpr.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_ravenpack.py",
        ],
        "clean": [],
    }

def task_link():
    """Link datasets together through shared identifiers"""
    yield {
        "name": "link_ravenpack_crsp",
        "doc": "Build crosswalk between RavenPack entity IDs and CRSP PERMNOs",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/link_ravenpack_crsp.py",
        ],
        "targets": [
            DATA_DIR / "raven_crsp_crosswalk.parquet",
            DATA_DIR / "ravenpack_djpr_with_permno.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/link_ravenpack_crsp.py",
            DATA_DIR / "ravenpack_djpr.parquet",
        ],
        "clean": [],
    }

def task_clean_data():
    """Clean and filter data"""
    yield {
        "name": "ravenpack",
        "doc": "Filter RavenPack headlines through the cleaning funnel",
        "actions": ["ipython ./src/clean_ravenpack.py"],
        "targets": [
            DATA_DIR / "ravenpack_clean.parquet",
            DATA_DIR / "ravenpack_stage_counts.json",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/clean_ravenpack.py",
            DATA_DIR / "ravenpack_djpr_with_permno.parquet",
            DATA_DIR / "CRSP_daily_stock.parquet",
        ],
        "clean": True,
    }

def task_labels():
    yield {
        "name": "labels",
        "doc": "Compute 3-day return window labels",
        "actions": ["ipython ./src/clean_labels.py"],
        "targets": [DATA_DIR / "labeled_dataset.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/clean_labels.py",
            DATA_DIR / "ravenpack_clean.parquet",
            DATA_DIR / "CRSP_daily_stock.parquet",
        ],
        "clean": True,
    }



def task_merge_sector():
    yield {
        "name": "merge_sector",
        "doc": "Merge GICS sector onto labeled headlines (point-in-time, S&P 500 only)",
        "actions": ["ipython ./src/merge_sector.py"],
        "targets": [DATA_DIR / "labeled_dataset_with_sector.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/merge_sector.py",
            DATA_DIR / "labeled_dataset.parquet",
            DATA_DIR / "sp500_constituents.parquet",
        ],
        "clean": True,
    }


def task_text_stats():
    yield {
        "name": "text_stats",
        "doc": "Compute headline text statistics (char/word/BERT token/Gemma token percentiles)",
        "actions": ["ipython ./src/compute_text_stats.py"],
        "targets": [DATA_DIR / "text_stats.json"],
        "file_dep": [
            "./src/settings.py",
            "./src/compute_text_stats.py",
            DATA_DIR / "labeled_dataset_with_sector.parquet",
        ],
        "clean": True,
    }


def _check_embed_chunks_exist(model_name):
    """Return a doit uptodate callable that checks if embedding chunks exist."""
    chunk_dir_map = {
        "bert": "embeddings_bert_chunks",
        "gemma": "embeddings_gemma_chunks",
    }

    def _check():
        chunk_dir = DATA_DIR / chunk_dir_map[model_name]
        return chunk_dir.exists() and any(chunk_dir.glob("chunk_*.parquet"))

    return _check


def task_embed():
    """Generate text embeddings for headlines"""
    models = {
        "bert": "embed_bert",
        "gemma": "embed_gemma",
    }
    for model_name, script_name in models.items():
        yield {
            "name": model_name,
            "doc": f"Generate {model_name} embeddings",
            "actions": [f"ipython ./src/{script_name}.py"],
            "file_dep": [
                "./src/settings.py",
                f"./src/{script_name}.py",
                DATA_DIR / "labeled_dataset_with_sector.parquet",
            ],
            "uptodate": [_check_embed_chunks_exist(model_name)],
            "clean": True,
        }


def task_pull_sector_etfs():
    """Pull SPDR Select Sector ETF daily prices from yfinance."""
    return {
        "actions": ["ipython ./src/pull_sector_etfs.py"],
        "targets": [DATA_DIR / "sector_etfs.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_sector_etfs.py",
        ],
        "clean": True,
    }


def task_build_sector_panel():
    """Build per-(sector, day) sentiment + return panel for each model."""
    models = ["bert", "gemma"]
    for model_name in models:
        yield {
            "name": model_name,
            "doc": f"Build sector sentiment panel from {model_name} predictions",
            "actions": [f"ipython ./src/build_sector_panel.py --model {model_name}"],
            "targets": [DATA_DIR / f"sector_sentiment_panel_{model_name}.parquet"],
            "file_dep": [
                "./src/settings.py",
                "./src/build_sector_panel.py",
                "./src/merge_sector.py",
                DATA_DIR / "labeled_dataset_with_sector.parquet",
                DATA_DIR / f"rolling_predictions_{model_name}.parquet",
                DATA_DIR / "CRSP_daily_stock.parquet",
                DATA_DIR / "sp500_constituents.parquet",
                DATA_DIR / "sector_etfs.parquet",
            ],
            "clean": True,
        }


def task_train():
    """Train rolling sentiment models"""
    models = [
        # "tfidf",
        "bert",
        "gemma",
        # "openai_small",
    ]
    for model_name in models:
        yield {
            "name": model_name,
            "doc": f"Train rolling model on {model_name} embeddings",
            "actions": [f"ipython ./src/train_rolling_model.py {model_name}"],
            "targets": [
                DATA_DIR / f"rolling_results_{model_name}.json",
                DATA_DIR / f"rolling_predictions_{model_name}.parquet",
            ],
            "file_dep": [
                "./src/settings.py",
                "./src/train_rolling_model.py",
                DATA_DIR / "labeled_dataset_with_sector.parquet",
            ],
            "task_dep": [f"embed:{model_name}"],
            "clean": True,
        }


# notebook_tasks = {
#     "01_explore_data_ipynb": {
#         "path": "./src/01_explore_data_ipynb.py",
#         "file_dep": [
#             "./src/pull_fred.py",
#             "./src/pull_market_data.py",
#         ],
#         "targets": [],
#     },
#     "02_embeddings_demos_ipynb": {
#         "path": "./src/02_embeddings_demos_ipynb.py",
#         "file_dep": [],
#         "targets": [],
#     },
#     "03_methodology_ipynb": {
#         "path": "./src/03_methodology_ipynb.py",
#         "file_dep": [
#             "./src/train_rolling_model.py",
#             "./src/clean_labels.py",
#             DATA_DIR / "labeled_dataset_with_sector.parquet",
#         ],
#         "targets": [],
#     },
#     "04_results_ipynb": {
#         "path": "./src/04_results_ipynb.py",
#         "file_dep": [
#             "./src/train_rolling_model.py",
#             DATA_DIR / "ravenpack_stage_counts.json",
#             DATA_DIR / "ravenpack_clean.parquet",
#             DATA_DIR / "text_stats.json",
#         ],
#         "targets": [],
#     },
# }


# # fmt: off
# def task_run_notebooks():
#     """Preps the notebooks for presentation format.
#     Execute notebooks if the script version of it has been changed.
#     """
#     for notebook in notebook_tasks.keys():
#         pyfile_path = Path(notebook_tasks[notebook]["path"])
#         notebook_path = pyfile_path.with_suffix(".ipynb")
#         yield {
#             "name": notebook,
#             "actions": [
#                 """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
#                 f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
#                 jupyter_execute_notebook(notebook_path),
#                 jupyter_to_html(notebook_path),
#                 mv(notebook_path, OUTPUT_DIR),
#                 """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
#             ],
#             "file_dep": [
#                 pyfile_path,
#                 *notebook_tasks[notebook]["file_dep"],
#             ],
#             "targets": [
#                 OUTPUT_DIR / f"{notebook}.html",
#                 *notebook_tasks[notebook]["targets"],
#             ],
#             "clean": True,
#         }
# # fmt: on

# sphinx_targets = [
#     "./docs/index.html",
# ]


# def task_build_chartbook_site():
#     """Compile Sphinx Docs"""
#     notebook_scripts = [
#         Path(notebook_tasks[notebook]["path"])
#         for notebook in notebook_tasks.keys()
#     ]
#     file_dep = [
#         "./README.md",
#         "./chartbook.toml",
#         *notebook_scripts,
#     ]

#     return {
#         "actions": [
#             "chartbook build -f",
#         ],  # Use docs as build destination
#         "targets": sphinx_targets,
#         "file_dep": file_dep,
#         "task_dep": [
#             "run_notebooks",
#         ],
#         "clean": True,
#     }


# def task_run_pytest():
#     """Run pytest and produce a machine-readable test report"""
#     return {
#         "actions": [f"pytest --junitxml='{OUTPUT_DIR}'/pytest_report.xml"],
#         "targets": [OUTPUT_DIR / "pytest_report.xml"],
#         "file_dep": sorted(glob.glob("./src/*.py")),
#         "clean": True,
#     }
