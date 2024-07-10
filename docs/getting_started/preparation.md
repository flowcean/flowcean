# Preparation

Flowcean is a Python framework and as such, you need to setup a runtime environment to work with it.
This section will cover the essential steps with hints for the more advanced users and a step-by-step explanation for beginners.

## Git

Git is a version control system, which you will need if you want to work with the source code of Flowcean or with any other source code.
By now, Flowcean does not have a release on PyPi, i.e., you have to download the source code to use it.
The recommended way is to use git for this.

### Install on Windows

User on windows can download the latest version on [https://git-scm.com/downloads](https://git-scm.com/downloads).
Double-click on the installation file and keep the settings to their defaults (unless you know what you're doing).

After the installation, open a Powershell and you should be able to type `git` into it and see the help page:

```PS
> git 
Usage: git [-v | --version] [-h | --help] [-C <path>] [-c <name>=<value>]
        [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
        [-p | --paginate | -P | --no-pager] [--no-replace-objects] [--bare]
        [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
        [--config-env=<name>=<envvar>] <command> [<args>]
```
**TODO** Show results from an actual power shell

### Install on Linux

Git is shipped with the package manager of all common distributions and you can install it from the command line.
See [here](https://git-scm.com/download/linux) to get the command for your distribution, e.g.:

```bash
sudo apt-get install git  # On Debian/Ubuntu
sudo pacman -S git  # On Arch linux
```

## Python

Flowcean is written in Python and requires at least version 3.12.
To check the version of your Python installation, open a Powershell or command line and type

```bash
> python3 --version
Python 3.12.4
```

If the result looks like `Python 3.12.x` (where `x` might be any number), then you're settled and you can continue to the next section.

### Install on Windows

Nowadays, Windows offers to options for development.
First, you can use plain Windows, e.g., using the Powershell.
Second, there is the Windows Subsystem for Linux (WSL), which uses an integrated virtual machine for linux.
Although we would recommend going the second route, this might not be applicable for everyone.

#### Python on Powershell

First, you need to download the installer for Python. 
You will always find the latest version on [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
Once downloaded, double-click on the installer and follow the instructions; most, if not all, settings can be kept at their defaults.

As additional step, to be able to use virtual environments, you have to open a powershell in administrator mode and type

```PS
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Afterwards, it may be necessary to restart your computer.

Finally, you should be able to successfully check the installation as described above. 

#### Python on WSL

Open a powershell in administrator mode and run the command

```PS
wsl --install
```

which will install WSL and Ubuntu on your machine.
Afterwards, you have to restart your computer and then follow the instructions at [Install on Linux](#install-on-linux-1)

If you have an older version of Windows or prefer a step-by-step guide, please refer to [https://learn.microsoft.com/en-us/windows/wsl/install-manual](https://learn.microsoft.com/en-us/windows/wsl/install-manual)



### Install on Linux

Most linux distributions ship Python by default. 
However, the version might be different from what is needed for Flowcean (>= 3.12).
If you're on Debian/Ubuntu, you have to execute a sequence of commands:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt install python3.12
```

**TODO** Check if this works for "newer" versions

Please check your distributions documentation for installation instructions if you're not on Debian/Ubuntu.

Finally, you should be able to successfully check the installation as [described above](#python). 

## Virtual Environment

In almost all cases it is strongly recommended to use a virtual environment when working with Python.
Although creating a virtual environment is rather simple, there are a few things to consider.
It is advisable that you create the virtual environment at a place, where you can find it again.
Most commonly, this is directly in the project folder or a "central" place in your home directory (,e.g., `~/.venvs` or `C:\Users\%USER%\.venvs`).
Choosing a unique name for the virtual environment can be helpfull if you choose the second option.

But let's assume you want to create the virtual environment in your project folder.
And let's assume we have to create this project folder first:

```bash
mkdir MyProject 
cd MyProject
```

The next step is to create a virtual environment (here we chose the name `myvenv`)

```bash
python3.12 -m venv myvenv
```

To activate the virtual environment, you have to call the activation script.

```bash
. ./myvenv/bin/activate  # On Linux/WSL
.\myvenv\Scripts\activate.ps1  # On Powershell
```

If you get an error on Powershell, check the [guide](#python-on-powershell) again.

You prompt should have changed now

```bash
>  # before activation
(myvenv) >  # after activation
```

You can now continue to the [installation](installation.md) of Flowcean.

