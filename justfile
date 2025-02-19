set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

[no-exit-message]
check: check-uv check-pre-commit check-types check-deptry

check-uv:
  @echo "🚀 Checking lock file consistency with 'pyproject.toml'"
  @uv lock --locked

check-pre-commit:
  @echo "🚀 Checking code style: Running pre-commit"
  @uv run pre-commit run --all-files

check-types:
  @echo "🚀 Static type checking: Running pyright"
  @uv run --group examples pyright

check-deptry:
  @echo "🚀 Checking for obsolete dependencies: Running deptry"
  @uv run --group examples deptry src

test:
  @echo "🚀 Testing code: Running pytest"
  @uv run python -m pytest tests --cov --cov-config=pyproject.toml

docs:
  @echo "🚀 Building documentation: Running mkdocs"
  @uv run mkdocs build --strict

docs-serve:
  @echo "🚀 Serving documentation: Running mkdocs"
  @uv run mkdocs serve

examples: examples-alp examples-boiler examples-coffee_machine examples-failure_time_prediction examples-linear_data examples-one_tank examples-robot_localization_failure

examples-alp:
  @echo "🚀 Running example: Automatic Lashing Platform"
  @uv run --directory ./examples/automatic_lashing_platform/ run.py

examples-boiler:
  @echo "🚀 Running example: Boiler"
  @uv run --directory ./examples/boiler/ run.py

examples-coffee_machine:
  @echo "🚀 Running example: Coffee Machine"
  @uv run --directory ./examples/coffee_machine/ run.py

examples-failure_time_prediction:
  @echo "🚀 Running example: Failure Time Prediction"
  @uv run --directory ./examples/failure_time_prediction/ run.py

examples-linear_data:
  @echo "🚀 Running example: Linear Data"
  @uv run --directory ./examples/linear_data/ run.py

examples-one_tank:
  @echo "🚀 Running example: One Tank"
  @uv run --directory ./examples/one_tank/ run.py

examples-robot_localization_failure:
  @echo "🚀 Running example: Robot Localization Failure"
  @uv run --directory ./examples/robot_localization_failure/ run.py
