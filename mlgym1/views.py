import csv, io
import pandas as pd
import numpy as np
from django.shortcuts import render, get_object_or_404, redirect
from .models import MyUser, ThetaString
from .forms import MyUserForm, TrainingMethodForm
from .algorithms import *
from .Koustav_LR import *
from .linreg_normal import *
from .chandra_Linear_Regression import LinearRegression, linreg_predict
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from .signals import *

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
	session_ended.send(sender=MyUser, user=request.user)
	logout(request)
	return redirect('homepage')

@login_required
def choose_method(request):
	form = TrainingMethodForm(request.POST or None)
	if request.POST and form.is_valid():
			method= form.cleaned_data.get("method_field")
			if method[0]=="0":
				return redirect('homepage')
			elif method[0]=="ptron":
				return redirect('csv_upload_perceptron')
			elif method[0]=="logreg":
				return redirect('csv_upload_logreg')
			elif method[0]=="linreg_normal":
				return redirect('csv_upload_linreg_normal')
			elif method[0]=="linreg":
				return redirect('csv_upload_linreg')
	return render(request, 'mlgym1/choose_method.html', {'form':form})

@login_required
def upload_csv_train_perceptron(request):
	if request.method == "GET":
		return render(request, 'mlgym1/csv_upload_perceptron.html', {})

	csv_file=request.FILES['filename']
	if not csv_file.name.endswith('.csv'):
		return redirect("csv_upload_perceptron")

	db=pd.read_csv(csv_file)
	theta_str=perceptron(db)
	request.user.thetas.all().delete()
	theta_model=ThetaString()
	theta_model.name="theta_perceptron"
	theta_model.theta_string=theta_str
	theta_model.user=request.user
	theta_model.save()
	response= redirect('test_upload_perceptron')
	return response

@login_required
def upload_csv_test_perceptron(request):
	thetas=request.user.thetas.all()
	trained=True
	result_available=False
	result={'tp':0,'fp':0,'fn':0,'tn':0,'f1':0,'acc':0}
	if len(thetas)==0:
		trained=False
	else:
		#obtain theta in numpy form for further use
		theta=thetas.filter(name="theta_perceptron")
		if len(theta)==0 or len(theta)>1:
			trained=False
		else:
			theta = str_to_numpy(theta[0].theta_string)

	if request.method == "GET":
		return render(request, 'mlgym1/test_upload_perceptron.html', {'trained':trained,'result_available':result_available, 'result':result})
	if not trained:
		return redirect('csv_upload_perceptron')
	csv_file=request.FILES['filename']
	csv_file_actual=request.FILES['actual_y']
	if not csv_file.name.endswith('.csv') or not csv_file_actual.name.endswith('.csv'):
		return redirect('test_upload_perceptron')
	db=pd.read_csv(csv_file)
	db_y=pd.read_csv(csv_file_actual)
	try:
		pred=(perceptron_predict(db, theta))
		tp,fp,fn,tn,f1,acc=perceptron_accuracy(pred,db_y.to_numpy())
		result={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':f1,'acc':acc}
		result_available=True
	except:
		result_available=False
	return render(request, 'mlgym1/test_upload_perceptron.html', {'trained':trained, 'result':result, 'result_available':result_available})

@login_required
def upload_csv_train_logreg(request):
	if request.method == "GET":
		return render(request, 'mlgym1/csv_upload_logreg.html', {})

	csv_file1=request.FILES['file1']
	csv_file2=request.FILES['file2']
	lr=str(request.POST.get('learning_rate'))
	iters=str(request.POST.get('max_iterations'))
	lr=float(lr)
	iters=int(iters)
	if not csv_file1.name.endswith('.csv') or not csv_file1.name.endswith('.csv') :
		return redirect("csv_upload_logreg")

	db_x=pd.read_csv(csv_file1)
	db_y=pd.read_csv(csv_file2)
	params=logreg_train(db_x, db_y,iters,lr)
	request.user.thetas.all().delete()
	theta_w=ThetaString()
	theta_w.name="theta_logreg_w"
	theta_w.theta_string=numpy_to_str(params["w"])
	theta_w.user=request.user
	theta_w.save()
	theta_b=ThetaString()
	theta_b.name="theta_logreg_b"
	theta_b.theta_string=str(params["b"])
	theta_b.user=request.user
	theta_b.save()
	return redirect('test_upload_logreg')

@login_required
def upload_csv_test_logreg(request):
	thetas=request.user.thetas.all()
	trained=True
	result_available=False
	theta_w=thetas.filter(name="theta_logreg_w")
	theta_b=thetas.filter(name="theta_logreg_b")
	if len(theta_w)==0 or len(theta_b)==0:
		trained=False
	else:
		theta_w=str_to_numpy(theta_w[0].theta_string)
		theta_b=float(theta_b[0].theta_string)
	if request.method =="GET":
		return render(request, 'mlgym1/test_upload_logreg.html',{'trained':trained,'result_available':result_available})
	csv_file=request.FILES['filename']
	if not csv_file.name.endswith('.csv'):
		return redirect('test_upload_nn4')
	db=pd.read_csv(csv_file)
	try:
		params={"w":theta_w,"b" :theta_b}
		result=logreg_predict(db,params)
		result_str=numpy_to_str(result)
		result_available=True
	except:
		result_str=""
		result_available=False
	return render(request, 'mlgym1/test_upload_logreg.html', {'trained':trained,'result_available':result_available,'result_string':result_str})

@login_required
def upload_csv_train_linreg_normal(request):
	if request.method == "GET":
		return render(request, 'mlgym1/csv_upload_linreg_normal.html', {})

	csv_file=request.FILES['filename']
	if not csv_file.name.endswith('.csv'):
		return redirect("csv_upload_linreg_normal")

	db=pd.read_csv(csv_file)
	theta=linreg_normal_train(db.iloc[:,:-1],db.iloc[:,-1])
	theta_str=numpy_to_str(theta)
	request.user.thetas.all().delete()
	theta_model=ThetaString()
	theta_model.name="theta_linreg_normal"
	theta_model.theta_string=theta_str
	theta_model.user=request.user
	theta_model.save()
	response= redirect('test_upload_linreg_normal')
	return response

@login_required
def upload_csv_test_linreg_normal(request):
	thetas=request.user.thetas.all()
	trained=True
	result_available=False
	if len(thetas)==0:
		trained=False
	else:
		#obtain theta in numpy form for further use
		theta=thetas.filter(name="theta_linreg_normal")
		if len(theta)==0 or len(theta)>1:
			trained=False
		else:
			theta = str_to_numpy(theta[0].theta_string)

	if request.method == "GET":
		return render(request, 'mlgym1/test_upload_linreg_normal.html', {'trained':trained,'result_available':result_available})
	if not trained:
		return redirect('csv_upload_linreg_normal')
	csv_file=request.FILES['filename']
	if not csv_file.name.endswith('.csv'):
		return redirect('test_upload_linreg_normal')
	db=pd.read_csv(csv_file)
	try:
		result_string=numpy_to_str(linreg_normal_predict(db, theta))
		result_available=True
	except:
		result_available=False
	return render(request, 'mlgym1/test_upload_linreg_normal.html', {'trained':trained, 'result':result_string, 'result_available':result_available})

@login_required
def upload_csv_train_linreg(request):
	if request.method == "GET":
		return render(request, 'mlgym1/csv_upload_linreg.html', {})

	csv_file=request.FILES['filename']
	lr=str(request.POST.get('learning_rate'))
	iters=str(request.POST.get('max_iterations'))
	term=str(request.POST.get('termination_cost'))
	reg=str(request.POST.get('regularization'))
	lr=float(lr)
	iters=int(iters)
	term=float(term)
	reg=float(reg)
	if not csv_file.name.endswith('.csv'):
		return redirect("csv_upload_linreg_normal")

	db=pd.read_csv(csv_file)
	theta=LinearRegression(db.iloc[:,:-1],db.iloc[:,-1],lr,term,iters,reg)
	theta_str=numpy_to_str(theta)
	request.user.thetas.all().delete()
	theta_model=ThetaString()
	theta_model.name="theta_linreg"
	theta_model.theta_string=theta_str
	theta_model.user=request.user
	theta_model.save()
	response= redirect('test_upload_linreg')
	return response

@login_required
def upload_csv_test_linreg(request):
	thetas=request.user.thetas.all()
	trained=True
	result_available=False
	if len(thetas)==0:
		trained=False
	else:
		#obtain theta in numpy form for further use
		theta=thetas.filter(name="theta_linreg")
		if len(theta)==0 or len(theta)>1:
			trained=False
		else:
			theta = str_to_numpy(theta[0].theta_string)

	if request.method == "GET":
		return render(request, 'mlgym1/test_upload_linreg.html', {'trained':trained,'result_available':result_available})
	if not trained:
		return redirect('csv_upload_linreg')
	csv_file=request.FILES['filename']
	if not csv_file.name.endswith('.csv'):
		return redirect('test_upload_linreg_normal')
	db=pd.read_csv(csv_file)
	try:
		result_string=numpy_to_str(linreg_predict(db, theta))
		result_available=True
	except:
		result_available=False
	return render(request, 'mlgym1/test_upload_linreg.html', {'trained':trained, 'result':result_string, 'result_available':result_available})
