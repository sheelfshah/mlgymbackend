from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from .signals import *
# Create your models here.

class MyUser(models.Model):
	user=models.OneToOneField(User, on_delete=models.CASCADE)

	def __str__(self):
		return self.user.username

class ThetaString(models.Model):
	name=models.CharField(max_length=200, default="")
	theta_string=models.TextField()
	user=models.ForeignKey(User, on_delete=models.CASCADE, related_name="thetas")

	def __str__(self):
		return self.name