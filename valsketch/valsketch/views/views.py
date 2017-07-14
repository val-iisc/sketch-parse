from flask import (Blueprint, render_template, redirect, flash, request, url_for, current_app)
from valsketch.forms import SigninForm, SignupForm
from valsketch.models import User, db, Annotation
from  valsketch.helper import file_list, part_list
import numpy.random as random
import os
import json
from jinja2 import Markup
import randomcolor
from flask_login import login_required, logout_user, login_user, current_user


views = Blueprint('views', __name__, url_prefix='')
rand_color = randomcolor.RandomColor();


@views.route('/')
def index():
    return render_template('index.html', active='index');

@views.route('/signin', methods=['GET', 'POST'])
def signin():
    form=SigninForm()
    if form.validate_on_submit():
        user = form.user
        login_user(user)
        return redirect(url_for('views.dashboard'))
    return render_template('signin.html', form=form, active='signin');

@views.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', active='dashboard')

@views.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        user = form.user
        user.last_idx = getIdx()
        db.session.add(user)
        db.session.commit()
        flash('Thanks for registering')
        return redirect(url_for('views.signin'))
    return render_template('signup.html', active='signup', form=form);

@views.route('/tool', methods=['GET', 'POST'])
@login_required
def tool():
    if current_user.annotations.count() >= current_user.annotation_limit:
        redirect(url_for('views.logout'))
    if request.method == 'POST':
        annotation = process_request(request.form)
        current_user.last_idx = current_user.last_idx + 1
        db.session.add(annotation)
        db.session.commit()
    sketch_file = file_list[int(current_user.last_idx)];
    sketch_meta, colors = get_sketch_meta(sketch_file)
    return render_template('tool.html', sketch_meta=sketch_meta, colors=colors, sketch_file=sketch_file)

@views.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('views.index'))


def get_sketch_meta(fname):
    fname = fname.strip();
    cat_name = os.path.dirname(fname)
    with open(os.path.join(current_app.root_path, 'static/sketches_svg', fname)) as sketch_file:
        svg = sketch_file.read();
        colors = rand_color.generate(count = len(part_list[cat_name]))

    return ({'parts' : part_list[cat_name],
            'svg'   : Markup(svg),
            'category': cat_name.capitalize()}, colors)

def process_request(user_response):
    annotation = Annotation(
            user_response['sketch_file'],
            user_response['sketch_category'],
            current_user.id,
            user_response['svg_part_paths'],
            user_response['part_paths'])
    return annotation

def getIdx():
    idx_file = os.path.join(current_app.root_path, 'static/idx.txt')
    with open(idx_file, 'r') as f:
        lidx = int(f.read())

    with(open(idx_file, 'w')) as f:
        f.write(str(lidx + 30))

    return lidx





# fb client id: 853753161392782
# fb client secret: 72d8b0c67dacfe2f2fc415c3137758e1


#@app.route('/')
#def index():
#    return render_template('index.html', active='index');
    #if not google.authorized:
    #    return redirect(url_for("google.login"))
    #resp = google.get("plus/v1/people/me")
    #assert resp.ok, resp.text
    #return "You are {email} on Google".format(email=resp.json()["emails"][0]["value"])

#app.secret_key = "supersekrit"
#blueprint = make_google_blueprint(
#    client_id="1096611460311-1fv15mis6p6sl6utj4pc5tcgl120hhup.apps.googleusercontent.com",
#    client_secret="sRixDroNfhH9RWPNk4gYjhMu",
#    scope=["profile", "email"]
#)
#app.register_blueprint(blueprint, url_prefix="/login")
