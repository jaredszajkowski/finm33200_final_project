"""
Load project configurations from .env files and the environment variables below.

Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `settings.py` is run on its own, it will create the appropriate
directories.

For information about the rationale behind decouple and this module,
see https://pypi.org/project/python-decouple/

Note that decouple mentions that it will help to ensure that
the project has "only one configuration module to rule all your instances."
This is achieved by putting all the configuration into the `.env` file.
You can have different sets of variables for difference instances,
such as `.env.development` or `.env.production`. You would only
need to copy over the settings from one into `.env` to switch
over to the other configuration, for example.
"""

from datetime import datetime
from decouple import RepositoryEnv
from pathlib import Path
from platform import system

########################################################
## Helper functions
########################################################


# OS type
def get_os():
    os_name = system()
    if os_name == "Windows":
        return "windows"
    elif os_name == "Darwin":
        return "nix"
    elif os_name == "Linux":
        return "nix"
    else:
        return "unknown"


## File paths
def if_relative_make_abs(path):
    """If a relative path is given, make it absolute, assuming
    that it is relative to the project root directory (BASE_DIR)

    Example
    -------
    ```
    >>> if_relative_make_abs(Path('_data'))
    WindowsPath('C:/Users/jdoe/GitRepositories/cookiecutter_chartbook/_data')

    >>> if_relative_make_abs(Path("C:/Users/jdoe/GitRepositories/cookiecutter_chartbook/_output"))
    WindowsPath('C:/Users/jdoe/GitRepositories/cookiecutter_chartbook/_output')
    ```
    """
    path = Path(path)
    if path.is_absolute():
        abs_path = path.resolve()
    else:
        abs_path = (defaults["BASE_DIR"] / path).resolve()
    return abs_path


########################################################
## Define defaults dictionary and load .env file
########################################################

# Absolute path to root directory of the project
BASE_DIR = Path(__file__).absolute().parent.parent

defaults = {
    "BASE_DIR": BASE_DIR,
    "DATA_DIR": if_relative_make_abs(BASE_DIR / "_data"),
    "MANUAL_DATA_DIR": if_relative_make_abs(BASE_DIR / "data_manual"),
    "OUTPUT_DIR": if_relative_make_abs(BASE_DIR / "_output"),
    "SOURCE_DIR": if_relative_make_abs(BASE_DIR / "src"),
    "OS_TYPE": get_os(),
    "START_DATE": datetime.strptime("2000-01-01", "%Y-%m-%d"),
    "END_DATE": datetime.strptime("2025-12-31", "%Y-%m-%d"),
    "HF_TOKEN": None,
}

# Load .env file and append to defaults
env_file = RepositoryEnv(BASE_DIR / ".env")

# Append each key-value pair from the .env file to the defaults dictionary
for key, value in env_file.data.items():
    defaults[key] = value


def config(
    var_name,
    default=None,
    cast=None,
):
    if var_name in defaults:
        var = defaults[var_name]
        if default is not None:
            raise ValueError(
                f"Default for {var_name} already exists. Check your settings.py file."
            )
        if cast is not None:
            # Allows for re-emphasizing the type of the variable
            # but does not allow for changing the type of the variable
            # if the variable is defined in the settings.py file
            if type(cast(var)) is not type(var):
                raise ValueError(
                    f"Type for {var_name} is already set. Check your settings.py file."
                )
    else:
        # If the variable is not defined in the settings.py file, raise an error
        raise Exception(
            f"{var_name} is not defined in settings.py. Please add it to the settings.py file."
        )
    return var


def create_directories():
    config("DATA_DIR").mkdir(parents=True, exist_ok=True)
    config("MANUAL_DATA_DIR").mkdir(parents=True, exist_ok=True)
    config("OUTPUT_DIR").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
