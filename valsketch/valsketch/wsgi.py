from valsketch import create_app

app = create_app('valsketch.config.ProductionConfig')
app.testing = True;
