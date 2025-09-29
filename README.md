# torched-tacaw
TACAW engine built around Hamish Brown's py_multislice. Running on torch, gpu accelaerated.

## Installation

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

## Project file structure

torched_tacaw/
│
├── torched_tacaw/              # Main package directory (contains your code)
│   ├── __init__.py             # Makes it a package; can import modules here
│   ├── tacaw_core.py
│   ├── tacaw_postprocessing.py
│   ├── units.py
│   ├── tools.py
│   └── coordinates.py
│
├── tests/                      # Optional: for your tests
│   └── test_tacaw_core.py      # Example test file
│
├── pyproject.toml              # Modern Python packaging config
├── README.md                   # Docs
├── LICENSE                     # License file
└── .gitignore                  # Ignore build, venv, etc.
