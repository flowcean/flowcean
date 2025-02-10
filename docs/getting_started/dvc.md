# DVC — Data Version Control

DVC is an open-source git-based version control tool to manage data.
We use DVC to manage experiment data.

DVC in this repository is set up to access the project files on the remote server from inside the VPN of the Institute of Logistics Engineering (ITL).

To pull all data from the remote server, use the following command:

```sh
dvc pull
```

## Writing to DVC Remote

To be able to write (push) to the remote server, you have to set the username and password for the user with write permissions.
Ask the AGenC team for access to the credentials.

```sh
dvc remote modify --local itl-nas user <xxxxxxxxx>
dvc remote modify --local itl-nas password <xxxxxxxxx>
```

Afterward, you can push using DVC.
Refer to the official [DVC documentation](https://dvc.org/doc) on how to use DVC for data management.
