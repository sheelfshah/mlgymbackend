import csv, io
import pandas as pd
import numpy as np
from django.shortcuts import render, get_object_or_404, redirect
from .models import MyUser
from .forms import MyUserForm
from .algorithms import perceptron, perceptron_predict
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout

# Create your views here.
def home(request):
	return render(request, 'mlgym1/homepage.html',{})

def register(request):
	if request.method=="POST":
		user_form=MyUserForm(data=request.POST)
		if user_form.is_valid():
			user=user_form.save()
			user.set_password(user.password)
			user.save()
			login(request, user)
			return redirect('homepage')
	else:
		user_form=MyUserForm()
	return render(request, 'mlgym1/register.html', {'user_form':user_form})

def login_user(request):
	if(request.method=="POST"):
		username = request.POST.get('username')
		password = request.POST.get('password')
		valid=authenticate(username=username, password=password)
		if valid:
			login(request, valid)
			return redirect('homepage')
		else:
			return redirect('login_false')
	else:
		form=MyUserForm()
		return render(request, 'mlgym1/login.html',{'user_form':form})

def false_login(request):
	return render(request, 'mlgym1/false_login.html',{})

@login_required
def logout_user(request):
	logout(request)
	return redirect('homepage')

@login_required
def upload_csv_train(request):
	if request.method == "GET":
		return render(request, 'mlgym1/csv_upload.html', {})

	csv_file=request.FILES['filename']
	if not csv_file.name.endswith('.csv'):
		return redirect("csv_upload")

	db=pd.read_csv(csv_file)
	theta=perceptron(db)
	response= redirect('test_upload')
	response.set_cookie('theta', theta)
	return response

@login_required
def upload_csv_test(request):
	theta_string=request.COOKIES.get('theta',"")
	trained=True
	result_available=False
	if theta_string=="":
		trained=False
	else:
		#obtain theta in numpy form for further use
		theta = np.array([[float(j) for j in i.split('\t')] for i in theta_string.splitlines()])

	if request.method == "GET":
		return render(request, 'mlgym1/test_upload.html', {'trained':trained,'result_available':result_available})
	if not trained:
		return redirect('csv_upload')
	csv_file=request.FILES['filename']
	if not csv_file.name.endswith('.csv'):
		return redirect('test_upload')
	db=pd.read_csv(csv_file)
	result_string=perceptron_predict(db, theta)
	result_available=True
	return render(request, 'mlgym1/test_upload.html', {'trained':trained, 'result':result_string, 'result_available':result_available})
