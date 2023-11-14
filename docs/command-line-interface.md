# Command Line Interface

The `agenc` command is a command-line tool used to interact with the AGenC framework. It provides a range of options for managing experiments and runtime configurations. This document will guide you through the usage of the `agenc` command and its available options.

## Prerequisites

Before using the `agenc` command, ensure that you have AGenC installed and set up correctly. You can obtain the AGenC framework and installation instructions from the [installation](installation.md) section.

## Usage

To invoke the `agenc` command, open your command prompt or terminal and navigate to root of the experiment directory. Then, you can use the following syntax:

```bash
agenc [-h] --experiment EXPERIMENT [--configuration CONFIGURATION] [--verbose]
```

### Example

Here's an example of how to use the `agenc` command to run an experiment:

```bash
agenc --experiment experiment.yaml --configuration configuration.yaml --verbose
```

In this example, we:

1. Specify the path to the experiment file using `--experiment`.
2. Optionally specify the path to the runtime configuration file using `--configuration`.
3. Increase verbosity using `--verbose` to get more detailed output.

Feel free to adjust the paths and options according to your specific AGenC setup and requirements.

Further examples can be found in the [examples](examples/index.md) section.
