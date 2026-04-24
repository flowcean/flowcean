import os

# Real PySR/JuliaCall tests segfault under pytest with the default
# auto-threaded Julia runtime. Pinning JuliaCall to one thread keeps the test
# process stable without changing normal Flowcean runtime behavior.
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
