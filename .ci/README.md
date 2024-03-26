# Container for Continuous Integration (CI)

Login to registry (requires a connection to TUHH network):

```bash
buildah login collaborating.tuhh.de:5005
```

Build the image:

```bash
buildah build --pull --tag collaborating.tuhh.de:5005/w-6/forschung/agenc/flowcean/ci .
```

Upload to registry:

```bash
buildah push collaborating.tuhh.de:5005/w-6/forschung/agenc/flowcean/ci
```
