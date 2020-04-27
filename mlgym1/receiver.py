from django.dispatch import receiver
from .signals import *
from . models import MyUser

@receiver(session_ended)
def delete_thetas(sender,user, **kwargs):
	if sender==MyUser:
		user.thetas.all().delete()