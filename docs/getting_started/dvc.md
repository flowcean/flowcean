# DVC — Data Version Control

DVC is an open-source git-based version control tool to manage data.
We use DVC to manage experiment data.

DVC independently pushes and pulls data to a remote server.
To be able to access the configured default remote server, you have to set the password.

```sh
dvc remote modify --local tuhh-cloud password <xxxxxxxxx>
```

Afterward, you can push and pull using DVC.
Refer to the official [DVC documentation](https://dvc.org/doc) on how to use DVC for data management.

To pull all data from the remote server, use the following command:

```sh
dvc pull
```
