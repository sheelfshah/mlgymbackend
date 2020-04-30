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
	("nn4","4 Layer Neural Network"),
	("0", "Don't train"),
	("logreg","Logistic Regression"),
	("linreg_normal","Linear Regression via Normal Equation")
	)
class TrainingMethodForm(forms.Form):
	method_field=forms.MultipleChoiceField(choices=training_methods)