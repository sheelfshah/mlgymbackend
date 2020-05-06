from django import forms
from django.contrib.auth.models import User
from .models import MyUser

class MyUserForm(forms.ModelForm):
	password = forms.CharField(widget=forms.PasswordInput)
	class Meta:
		model=User
		fields={'username','password'}

training_methods=(
	("ptron","Perceptron"),
	("logreg","Logistic Regression"),
	("linreg_normal","Linear Regression via Normal Equation"),
	("linreg","Linear Regression via Gradient Descent"),
	("0", "Be savage and don't train")
	)
class TrainingMethodForm(forms.Form):
	method_field=forms.MultipleChoiceField(choices=training_methods)