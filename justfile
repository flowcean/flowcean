setup:
  @echo "🚀 Setting up development environment"
  @uv sync
  @uv run pre-commit install

check:
  @echo "🚀 Checking lock file consistency with 'pyproject.toml'"
  @uv lock --locked
  @echo "🚀 Checking code style: Running pre-commit"
  @uv run pre-commit run --all-files
  @echo "🚀 Static type checking: Running pyright"
  @uv run pyright
  @echo "🚀 Checking for obsolete dependencies: Running deptry"
  @uv run deptry src

test:
  @echo "🚀 Testing code: Running pytest"
  @uv run python -m pytest --cov --cov-config=pyproject.toml

docs:
  @echo "🚀 Building documentation: Running mkdocs"
  @uv run mkdocs build --strict

docs-serve:
  @echo "🚀 Serving documentation: Running mkdocs"
  @uv run mkdocs serve

examples: examples-alp examples-boiler examples-coffee_machine examples-failure_time_prediction examples-linear_data examples-one_tank examples-ros_offline

examples-alp:
  @echo "🚀 Running example: Automatic Lashing Platform"
  @uv run --directory ./examples/automatic_lashing_platform/ --with-editable ../../ run.py

examples-boiler:
  @echo "🚀 Running example: Boiler"
  @uv run --directory ./examples/boiler/ --with-editable ../../ run.py

examples-coffee_machine:
  @echo "🚀 Running example: Coffee Machine"
  @uv run --directory ./examples/coffee_machine/ --with-editable ../../ run.py

examples-failure_time_prediction:
  @echo "🚀 Running example: Failure Time Prediction"
  @uv run --directory ./examples/failure_time_prediction/ --with-editable ../../ run.py

examples-linear_data:
  @echo "🚀 Running example: Linear Data"
  @uv run --directory ./examples/linear_data/ --with-editable ../../ run.py

examples-one_tank:
  @echo "🚀 Running example: One Tank"
  @uv run --directory ./examples/one_tank/ --with-editable ../../ run.py

examples-ros_offline:
  @echo "🚀 Running example: ROS Offline"
  @uv run --directory ./examples/ros_offline/ --with-editable ../../ run.py
