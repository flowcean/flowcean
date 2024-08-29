# Installation

First, you have to download the source code of Flowcean (a package at PyPi will follow in the future).
You can either use Git as recommended in the [Preparation](preparation.md) or download it as zip file from [Github](https://github.com/flowcean/flowcean).

If you followed the previous guide, you should have a folder `MyProject` which contains the virtual environment `myvenv`.
Put the downloaded sources into the folder `MyProject` or execute from within that folder:

```bash
git clone https://github.com/flowcean/flowcean.git
cd flowcean
```

Your folder structure should now look like this:

```
|--MyProject/
|  |--flowcean/
|  |  |--docs/
|  |  |--src/
|  |  |--tests/
|  |  |--pyproject.toml
|  |  |--<other source files>
|  |--myvenv/
```

It is recommended to use `flowcean` in a virtual environment.
Make sure your virtual environment is [activated](preparation.md#virtual-environment) or consult the official [python documentation](https://docs.python.org/3/library/venv.html) for further information.

Now you can install the Flowcean package (assuming that your current working directory is inside the `flowcean` folder):

```sh
pip install -e .
```

For the full functioning experience, Flowcean splits its features into optional dependency packages.
Have a look at `pyproject.toml` and respectively install additional dependency groups.
E.g., to get everything from Flowcean, you can install it with `all` additional features:

```bash
pip install -e .[all]
```