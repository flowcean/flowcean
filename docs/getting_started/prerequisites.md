# Prerequisites

Before using and developing `flowcean`, ensure you have the necessary tools and environments set up.
This page outlines the requirements and how to install them.

## Git

Git is a widely used version control system that will help you manage the source code and dependencies.
Follow the steps below to download and install Git or consult the [official documentation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

=== "Linux"

    Most Linux distributions have Git available in their package managers.
    You can install it by running one of the following commands, depending on your distribution:

      ```bash
      # Debian-based distributions like Ubuntu
      sudo apt-get install git
      # Fedora
      sudo dnf install git
      # Arch Linux
      sudo pacman -S git
      ```

=== "macOS"

    You can install Git using a package manager like Homebrew.
    If you have Homebrew installed, simply run:

      ```bash
      brew install git
      ```

=== "Windows"

    Visit the official [Git website](https://git-scm.com/downloads) and download the appropriate version of Git for your operating system.
    Run the downloaded `.exe` file and follow the installation instructions.
    You might want to select the default options unless you have specific needs.

Verify the installation by checking the version:

```bash
git --version
```

This command should display the installed version of Git.

## uv

`uv` is a package manager and environment management tool that eases the handling of Python projects.
Install `uv` if you don't already have it:

=== "Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

See the [installation documentation](https://docs.astral.sh/uv/getting-started/installation/) for details and alternative installation methods.

## Python

While `uv` manages the necessary Python interpreters for you, you might want to install Python manually if you don't intend to use `uv`.

### Manual Python Installation

=== "Linux"

    Most Linux distributions come with Python pre-installed.
    However, you can also install or update Python using your package manager:

    - For Debian-based distributions (like Ubuntu):

      ```bash
      sudo apt-get update
      sudo apt-get install python3
      ```

    - For Fedora:

      ```bash
      sudo dnf install python3
      ```

    - For Arch Linux:

      ```bash
      sudo pacman -S python
      ```

=== "macOS"

    - Install Python using Homebrew by running:

      ```bash
      brew install python
      ```

    - Alternatively, download the installer from the official Python website and follow the instructions.

=== "Windows"

    1. Download the latest version of Python from the official Python website.
    2. Run the downloaded installer. Be sure to check the "Add Python to PATH" option before installing.
    3. Follow the installation prompts to complete the process.
    4. To use virtual environments in Powershell, you have to open it once in administrator mode and type

      ```PS
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
      ```

    5. You may need to restart your computer.

=== "Windows (WSL)"

    1. Open a powershell in administrator mode and run the command

      ```PS
      wsl --install
      ```

    2. Follow the instructions, for step-by-step guide, refer to the [official documentation](https://learn.microsoft.com/en-us/windows/wsl/install-manual)

    3. Now follow the instructions for Linux


Verify the installation by checking the Python version:

```bash
python3 --version
```

or, if `python3` is unavailable, try:

```bash
python --version
```

This command should display the installed version of Python.

## Just

[just](https://github.com/casey/just) is a handy way to save and run project-specific commands.
Follow the steps below to download and install `just` as a pre-built binary or consult the [official installation guide](https://github.com/casey/just?tab=readme-ov-file#installation).

You can use the following command on Linux, MacOS, or Windows to download the latest release, just replace `<DEST>` with the directory where you'd like to put `just`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to <DEST>
```

For example, to install `just` to `~/.local/bin`:

```bash
# create ~/.local/bin
mkdir -p ~/.local/bin

# download and extract just to ~/.local/bin/just
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin

# add `~/.local/bin` to the paths that your shell searches for executables
# this line should be added to your shells initialization file,
# e.g. `~/.bashrc` or `~/.zshrc`
export PATH="$PATH:$HOME/.local/bin"

# just should now be executable
just --help
```
