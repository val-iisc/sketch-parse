


## Instructions

### Setup the database

We use the sqlite database system which basically stores the database in a single file (and doesn't follow the client-server model).
The data is stored in `valsketch/testing.db` which must be created by the user. Then, to create the relevant tables within the
file:

```
python manager.py createdb
```

This would ensure that the database is now ready for use to store the sketch annotations.


### Setup python environment

We recommend the use of virtual environments. Once a virtual environment is setup and activated, you should install all packages in listed 
in the `requirements.txt` file by `pip install -r requirements.txt`. This command can be executed even outside a virtual environment, but be aware
that it will install these python packages globally and may require admin privileges.


### Setup the per-user file limit

Since the tool was made to crowdsource sketch annotations, each user of the tool is limited to a fixed number of sketches to annotate. This number
can be specified in the `valsketch/static/idx.txt` file.


### Setup part names of objects

This annotator was made to annotate parts of sketches of different objects. For each object, a number of parts are listed in `valsketch/static/parts.json`
file. You may edit the file to suit your needs. Note that the object name that is used in the `parts.json` file must match the directory name which stores
sketches of that particular object.

### Add sketches

All sketches must be stored in suitably named directories in `valsketch/static/sketches_svg` directory. The annotator tool only works with `.svg` files so make
sure that sketches are of this format.
