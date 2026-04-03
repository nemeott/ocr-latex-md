Create a Python 3.13 virtual environment with `uv` [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/):

# Setup

```bash
uv venv --python 3.13
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

# Development

Add dependencies with:

```bash
uv pip install <package-name>
```

Update dependencies file with:

```bash
# List the installed packages and their versions
uv pip freeze
# Manually update

# Or redirect the output to the requirements.txt file
uv pip freeze > requirements.txt
```
