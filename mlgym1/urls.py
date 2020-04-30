from django.urls import path
from . import views

urlpatterns=[
	path('', views.home, name="homepage"),
	path('login/', views.login_user, name='login'),
	path('logout/', views.logout_user, name='logout'),
	path('register/', views.register, name='register'),
	path('login_retry/', views.false_login, name="login_false"),
	path('method/',views.choose_method, name="method_choice"),
	path('perceptron/train/', views.upload_csv_train_perceptron, name="csv_upload_perceptron"),
	path('perceptron/test/', views.upload_csv_test_perceptron, name="test_upload_perceptron"),
	path('nn4/train', views.upload_csv_train_nn4, name="csv_upload_nn4"),
	path('nn4/test', views.upload_csv_test_nn4, name="test_upload_nn4"),
	path('logreg/train',views.upload_csv_train_logreg, name="csv_upload_logreg"),
	path('logreg/test', views.upload_csv_test_logreg, name="test_upload_logreg"),
	path('normallinreg/train', views.upload_csv_train_linreg_normal, name="csv_upload_linreg_normal"),
	path('normallinreg/test', views.upload_csv_test_linreg_normal, name="test_upload_linreg_normal"),
]