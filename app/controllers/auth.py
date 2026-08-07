from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import User, Patient, Doctor
from app.forms.auth import LoginForm, RegistrationForm

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle user login.
    GET: Display login form
    POST: Process login credentials
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = form.user
        login_user(user, remember=True)
        flash(f'Welcome back, {user.full_name}!', 'success')
        
        # Redirect to the page user was trying to access
        next_page = request.args.get('next')
        return redirect(next_page) if next_page else redirect(url_for('main.index'))
    
    return render_template('auth/login.html', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handle user registration.
    GET: Display registration form
    POST: Process registration data and create Patient or Doctor based on role
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Create appropriate user type based on role
        if form.role.data == 'patient':
            user = Patient(
                username=form.username.data,
                email=form.email.data,
                full_name=form.full_name.data,
                role='patient',
                age=form.age.data,
                gender=form.gender.data,
                phone=form.phone.data,
                address=form.address.data,
                district=form.district.data if form.district.data else 'Erode'
            )
        elif form.role.data == 'doctor':
            user = Doctor(
                username=form.username.data,
                email=form.email.data,
                full_name=form.full_name.data,
                role='doctor',
                phone=form.phone.data
            )
        else:
            user = User(
                username=form.username.data,
                email=form.email.data,
                full_name=form.full_name.data,
                role='admin'
            )
        
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        flash(f'Registration successful! Welcome, {user.full_name}!', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    """
    Handle user logout.
    Requires user to be logged in.
    """
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('main.index'))

@auth_bp.route('/profile')
@login_required
def profile():
    """
    Display user profile.
    Requires user to be logged in.
    """
    return render_template('auth/profile.html', user=current_user)
