from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
# Create your models here.

class MyUser(models.Model):
	user=models.OneToOneField(User, on_delete=models.CASCADE)