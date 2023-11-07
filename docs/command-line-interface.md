# Command Line Interface

The `agenc` command is a command-line tool used to interact with the AGenC framework. It provides a range of options for managing experiments and runtime configurations. This document will guide you through the usage of the `agenc` command and its available options.

## Prerequisites

Before using the `agenc` command, ensure that you have AGenC installed and set up correctly. You can obtain the AGenC framework and installation instructions from the [installation](installation.md) section.

## Usage

To invoke the `agenc` command, open your command prompt or terminal and navigate to the directory containing your AGenC installation. You can use the following syntax:

```bash
agenc [-h] --experiment EXPERIMENT [--configuration CONFIGURATION] [--verbose]
```

### Options

- `-h, --help`: Display the help message and exit. This option provides a summary of the available command-line options.

- `--experiment EXPERIMENT`: Specify the path to the experiment file. The experiment file is essential for defining the parameters and settings for your AGenC experiments, such as the desired learning algorithm or learning metrics. Make sure to provide the full file path.

- `--configuration CONFIGURATION`: Optional. Specify the path to the runtime configuration file. The runtime configuration file allows you to customize the behavior of AGenC during the experiment. If not provided, the default configuration will be used.

- `--verbose`: Optional. Use this option to increase the verbosity of the command. This will result in more detailed output, which can be helpful for debugging and understanding the inner workings of AGenC.

### Example

Here's an example of how to use the `agenc` command to run an experiment:

```bash
agenc --experiment C:\path\to\my_experiment.yaml --configuration C:\path\to\my_configuration.yaml --verbose
```

In this example, we:

1. Specify the path to the experiment file using `--experiment`.
2. Optionally specify the path to the runtime configuration file using `--configuration`.
3. Increase verbosity using `--verbose` to get more detailed output.

Feel free to adjust the paths and options according to your specific AGenC setup and requirements.

Further examples can be found in the [examples](examples/index.md) section.

## Conclusion

The `agenc` command-line tool is a powerful and flexible way to manage your AGenC experiments and configurations. Use the provided options to fine-tune your experiments and achieve your desired results. For more detailed information on AGenC and its capabilities, refer to the official documentation.