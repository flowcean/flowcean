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
  @echo "🚀 Static type checking: Running basedpyright"
  @uv run --all-packages --all-extras basedpyright

check-deptry:
  @echo "🚀 Checking for obsolete dependencies: Running deptry"
  @uv run deptry src

test:
  @echo "🚀 Testing code: Running pytest"
  @uv run python -m pytest tests --cov --cov-config=pyproject.toml

docs:
  @echo "🚀 Building documentation..."
  @echo "   - Running javadoc"
  @mvn javadoc:javadoc -q -f java/AutomataLearner/pom.xml
  @rm -rf docs/examples/java-automata/
  @mv java/AutomataLearner/target/site/* docs/examples/java-automata/
  @echo "   - Running mkdocs"
  @uv run mkdocs build --strict

docs-serve:
  @echo "🚀 Serving documentation: Running javadoc and mkdocs"
  @mvn javadoc:javadoc -f java/AutomataLearner/pom.xml
  @rm -rf docs/examples/java-automata/
  @mv java/AutomataLearner/target/site/* docs/examples/java-automata/
  @uv run mkdocs serve

examples: examples-alp examples-boiler examples-coffee_machine examples-linear_data examples-one_tank examples-robot_localization_failure examples-energy_system examples-xor

examples-alp:
  @echo "🚀 Running example: Automatic Lashing Platform"
  @uv run --directory ./examples/automatic_lashing_platform/ run.py

examples-boiler:
  @echo "🚀 Running example: Boiler"
  @uv run --directory ./examples/boiler/ run.py

examples-coffee_machine:
  @echo "🚀 Running example: Coffee Machine"
  @uv run --directory ./examples/coffee_machine/ run.py

examples-ddtig:
  @echo "🚀 Running example: DDTIG"
  @uv run --directory ./examples/ddtig/ run.py

examples-linear_data:
  @echo "🚀 Running example: Linear Data"
  @uv run --directory ./examples/linear_data/ run.py

examples-one_tank:
  @echo "🚀 Running example: One Tank"
  @uv run --directory ./examples/one_tank/ run_offline.py
  @uv run --directory ./examples/one_tank/ run_incremental.py

examples-robot_localization_failure:
  @echo "🚀 Running example: Robot Localization Failure"
  @uv run --directory ./examples/robot_localization_failure/ run.py

examples-energy_system:
  echo "🚀 Running example: Energy System"
  @uv run --directory ./examples/energy_system/ run_only_short.py

examples-xor:
  @echo "🚀 Running example: XOR"
  @uv run --directory ./examples/xor/ run.py

generate-proto:
  @echo "🚀 Generating Python and Java definitions from gRPC proto files"
  @uv run python -m grpc_tools.protoc --proto_path=. --python_out=. --mypy_out=. --grpc_python_out=. --mypy_grpc_out=. src/flowcean/grpc/proto/learner.proto
  @protoc --plugin=protoc-gen-grpc-java=src/flowcean/grpc/proto/protoc-gen-grpc-java --grpc-java_out=. --experimental_allow_proto3_optional=true ./src/flowcean/grpc/proto/learner.proto
  @protoc --plugin=protoc-gen-grpc-java=src/flowcean/grpc/proto/protoc-gen-grpc-java --java_out=. --experimental_allow_proto3_optional=true ./src/flowcean/grpc/proto/learner.proto
