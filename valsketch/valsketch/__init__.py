from flask import Flask, session, redirect, url_for, escape, render_template
from views import views
from jinja2 import Markup
from valsketch.models import db
from valsketch.extensions import (
    #debug_toolbar,
    login_manager
)

app = Flask('valsketch');

def create_app(object_name):

    app = Flask(__name__)
    app.config.from_object(object_name)


    ## initialize the debug tool bar
    #debug_toolbar.init_app(app)

    # initialize SQLAlchemy
    db.init_app(app)

    login_manager.init_app(app)

    app.register_blueprint(views)

    return app
