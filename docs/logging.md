# Logging

The `AGenC` framework makes extensive use of logging.
Each module has its own logger that can be enabled or disabled.

## Basic Usage

If the `agenc` command line tool is used, logging is enabled by default and everything logged with the level `INFO` or lower.

To increase verbosity, pass `--verbose` to the command line tool, e.g.,

```bash
agenc ---verbose --experiment <path/to/experiment-file>
```

will increase verbosity to level `DEBUG`.

## Advanced Configuration

Logging can be configured more fine-grained by using a runtime configuration file.
The logging configuration is specified under the `logging` key and follows the [dictionary schema](https://docs.python.org/3/library/logging.config.html#logging-config-dictschema) of the python logging library.
