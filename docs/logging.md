# Logging Configuration

The `AGenC` framework makes extensive use of logging.
Each module has its own logger that can be enabled or disabled.

## Basic Usage

If the `agenc` command line tool is used, logging is enabled by default and everything logged with the level WARNING or higher.

To increase verbosity, pass `--verbose` (or `-v` ) to the command line tool, e.g., 

```bash
agenc -vv --experiment <path/to/experiment-file>
```

will increase verbosity to level DEBUG.

**NOTE**: The default logging handler for the console only prints INFO messages (or higher). 
You will find the debug outputs in the default log file `agenc-debug.log` .

## Advanced Configuration

Logging can be configured more fine-grained by using a runtime configuration file.
To make use of it, **create** a file named `agenc-runtime-conf.yml` in the directory from which you start the `agenc` command (or see [here](##config-locations) where to store it persistently)

```yaml
%YAML 1.2
# Configuration loaded from: (DEFAULT)
---

logging:
  formatters:
    debug: {format: '%(asctime)s [%(name)s (%(process)d)][%(levelname)s] %(message)s
        (%(module)s.%(funcName)s in %(filename)s:%(lineno)d)'}
    simple: {format: '%(asctime)s [%(name)s][%(levelname)s] %(message)s'}
  handlers:
    console: {class: logging.StreamHandler, formatter: simple, level: INFO, stream: ext://sys.stdout}
    console-debug: {class: logging.StreamHandler, formatter: debug, level: DEBUG,
      stream: ext://sys.stdout}
    logfile: {class: logging.FileHandler, filename: agenc.log, formatter: simple,
      level: INFO, mode: w}
    logfile-debug: {class: logging.FileHandler, filename: agenc_debug.log, formatter: debug,
      level: DEBUG, mode: w}
  loggers:
    agenc: {level: WARNING}
  root:
    handlers: [console, logfile-debug]
    level: ERROR
  version: 1

```

### Loggers and Logging Levels

The most important field is `loggers`, which allows you to configure logging levels for individual loggers.
You could, e.g., add a specific configuration for `agenc.experiment` with level `INFO`.
The result would look like this

```yaml

logging:
  ...
  loggers: 
    agenc: {level: WARNING}
    agenc.experiment: {level: INFO}
  ...
```

### Logging Handlers

The field `handlers` defines, where the logging will be visible.
There are four handlers pre-configured: `console`, `console-debug`, `logfile`, and `logfile-debug`.
While `console` and `logfile` is rather obvious, the `-debug` version of those handlers have `DEBUG` as minimum logging level and they use a different *formatter*.

The `logfile` handlers allow to define the name of the logfile and the writing mode (`w` to create a new file each time and `a` to append to an existing file with that name).

### Logging Formatters

Two logging `formatters` are pre-configured.
The `simple` formatter prints out a timestamp, the logger name, the log level, and the log message.
The `debug` formatter adds process ID, module and function names, the file name, and the line of the logging statement.
This should help finding the root for log messages more easily.

### The Root Handler

The field `root` configures, which logging handlers to use and which logging level should be applied to **every other logger** that is not configured in the runtime config.

You can change the handlers as you like (e.g., from console to console-debug) but note that if you have more than one console-based logging handler, you will receive messages multiple times.

Setting the root level to WARNING, INFO, or DEBUG will produce lots of messages from all the libraries that are imported in the code.
**Change only if you really need all those log messages**

## Config Locations

The `AGenC` framework has a default runtime configuration, which will be used if no other configuration is available.
The framework will check at two different locations:

- user_config_dir
- current working directory

The actual locations differ depending on the operating system you're using.

For e.g., the user_config_dir is,

- `C:\Users\<username>\AppData\Local\agenc` on Windows and
- `/home/<username>/.config/agenc` on Linux
- `/Library/Application Support/agenc` on macOS

You can put the AGenC runtime configuration file in that directory.
It will be used regardless from where you start the programm.

There is a third option, if you want to use a specific runtime configuration, e.g., for debugging.
You can pass the path to a runtime configuration file to the `agenc` command:

```bash
agenc -c <path/to/runtime-conf.yml> 
```

