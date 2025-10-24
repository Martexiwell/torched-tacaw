# torched-tacaw
TACAW engine built around Hamish Brown's py_multislice. Running 
on torch, gpu accelerated.

## Installation

### Prep

In your virtual environment, it is a good idea to install torch 
and make sure that it is compatible with the current machine for 
it to work well with gpu. (This is relevant probably mostly for 
clusters specifically. Since it is not straightforward to install
environment in login node for a special node that you will use 
for your calculation)

### 1. Install py_multislice

Clone and install Hamish Brown's py_multislice:
```bash
git clone https://github.com/HamishGBrown/py_multislice.git
cd py_multislice
pip install -e .
```

### 2. Install torched_tacaw

Clone and install via pip:
```bash
git clone https://github.com/Martexiwell/torched_tacaw.git
pip install -e torched_tacaw
```

### Upgrade torched tacaw
when in torched_tacaw directory:
```bash
git pull                # pull all commits from remote for the current branch
git checkout <branch>   # changes current branch 
```

## Notes

- be mindful of the fence post problem - by convention we don't add the 
"+1" point in our scanning grid


## Project file structure
```
torched_tacaw/
│
├── torched_tacaw/              # Main package directory (contains your code)
│   ├── __init__.py             # Makes it a package; can import modules here
│   ├── core.py
│   ├── postprocessing.py
│   ├── units.py
│   ├── tools.py
│   ├── io.py
│   └── coordinates.py
│
├── tests/                      # Optional: for your tests
│   └── test_tacaw_core.py      # Example test file
│
├── examples/                   # s
│   └── tutorial/
│       ├── tutorial.md 
│       └── ...
│
├── pyproject.toml              # Modern Python packaging config
├── README.md                   # Docs
├── LICENSE                     # License file
└── .gitignore                  # Ignore build, venv, etc.

```

## git repository

remote repo hosted on https://github.com/Martexiwell/torched-tacaw

### branches
- **default branch:** main
- **development branch:** dev
  - other branches 
  - 