# torched-tacaw
TACAW engine built around Hamish Brown's py_multislice. Running on torch, gpu accelaerated.

## file structure
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
