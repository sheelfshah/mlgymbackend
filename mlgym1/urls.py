from django.urls import path
from . import views

urlpatterns=[
	path('', views.home, name="homepage"),
	path('login/', views.login_user, name='login'),
	path('logout/', views.logout_user, name='logout'),
	path('register/', views.register, name='register'),
	path('login_retry/', views.false_login, name="login_false"),
	path('upload/', views.upload_csv, name="csv_upload"),
]