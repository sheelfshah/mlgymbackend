import django.dispatch
session_ended = django.dispatch.Signal(providing_args=["user"])