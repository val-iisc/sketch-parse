import json, os

# recaptcha key: 6Ld1QggUAAAAAC-4NBw4BYesnQ_ykoYMnylT5WGA
# recaptcha secret: 6Ld1QggUAAAAABEHsa_1tnBQ3a7k1rSfl8_K698Y

class Config(object):
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///testing.db'
    WTF_CSRF_SECRET_KEY = 'app_forms_key'
    SECRET_KEY = 'secret-key'

class ProductionConfig(Config):
    RECAPTCHA_PUBLIC_KEY = '6Ld1QggUAAAAAC-4NBw4BYesnQ_ykoYMnylT5WGA'
    RECAPTCHA_PRIVATE_KEY = '6Ld1QggUAAAAABEHsa_1tnBQ3a7k1rSfl8_K698Y'

class DevelopmentConfig(Config):
    DEBUG = True
    RECAPTCHA_PUBLIC_KEY = '6Ld1QggUAAAAAC-4NBw4BYesnQ_ykoYMnylT5WGA'
    RECAPTCHA_PRIVATE_KEY = '6Ld1QggUAAAAABEHsa_1tnBQ3a7k1rSfl8_K698Y'

class TestingConfig(Config):
    TESTING = True
