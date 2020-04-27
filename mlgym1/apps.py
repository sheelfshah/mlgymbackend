from django.apps import AppConfig


class Mlgym1Config(AppConfig):
    name = 'mlgym1'
    def ready(self):
        from . import receiver    