# Container for Continuous Integration (CI)

Login to registry:

```bash
docker login collaborating.tuhh.de:5005
```

Build the image:

```bash
docker build -t collaborating.tuhh.de:5005/w-6/forschung/agenc/agenc/ci .
```

Upload to registry:

```bash
docker push collaborating.tuhh.de:5005/w-6/forschung/agenc/agenc/ci
```
