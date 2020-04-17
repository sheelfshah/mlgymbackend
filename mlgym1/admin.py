from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from mlgym1.models import MyUser
# Register your models here.

class  MyUserInline(admin.StackedInline):
	model =MyUser
	can_delete = False

class UserAdmin(BaseUserAdmin):
	inlines= [MyUserInline]

admin.site.unregister(User)
admin.site.register(User, UserAdmin)