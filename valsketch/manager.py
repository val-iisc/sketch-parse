#!/usr/bin/env python

import os

from flask_script import Manager, Server
from flask_script.commands import ShowUrls, Clean
from valsketch import create_app
from valsketch.models import db, User


env = os.environ.get('APPNAME_ENV', 'development')
app = create_app('valsketch.config.%sConfig' % env.capitalize())

manager = Manager(app)
manager.add_command("server", Server(threaded=True))
manager.add_command("show-urls", ShowUrls())
manager.add_command("clean", Clean())

@manager.shell
def make_shell_context():
    """ Creates a python REPL with several default imports
        in the context of the app
    """

    return dict(app=app, db=db, User=User)

@manager.command
def createdb():
    """ Creates a database with all of the tables defined in
        your SQLAlchemy models
    """
    db.create_all()

if __name__ == "__main__":
    manager.run()
