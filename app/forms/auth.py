from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, IntegerField, TextAreaField, ValidationError
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional
from app.models import User

class LoginForm(FlaskForm):
    """
    Login form for user authentication.
    Validates username/email and password credentials.
    """
    username = StringField('Username or Email', 
                         validators=[DataRequired(message='Username or email is required'),
                                   Length(min=3, max=120)])
    password = PasswordField('Password', 
                             validators=[DataRequired(message='Password is required'),
                                       Length(min=6, max=128)])
    
    def __init__(self, *args, **kwargs):
        super(LoginForm, self).__init__(*args, **kwargs)
        self.user = None
    
    def validate(self, extra_validators=None):
        """
        Validate login credentials.
        Checks if user exists and password matches.
        """
        if not super(LoginForm, self).validate(extra_validators):
            return False
        
        # Check if user exists by username or email
        user = User.query.filter(
            (User.username == self.username.data) | (User.email == self.username.data)
        ).first()
        
        if user is None:
            self.username.errors.append('Invalid username or email')
            return False
        
        if not user.check_password(self.password.data):
            self.password.errors.append('Invalid password')
            return False
        
        if not user.is_active:
            self.username.errors.append('Account is inactive')
            return False
        
        self.user = user
        return True

class RegistrationForm(FlaskForm):
    """
    Registration form for new user accounts.
    Collects user information with validation.
    """
    username = StringField('Username',
                         validators=[DataRequired(message='Username is required'),
                                   Length(min=3, max=80, message='Username must be 3-80 characters')])
    email = StringField('Email',
                       validators=[DataRequired(message='Email is required'),
                                 Email(message='Invalid email address'),
                                 Length(max=120)])
    full_name = StringField('Full Name',
                           validators=[DataRequired(message='Full name is required'),
                                     Length(min=2, max=200)])
    password = PasswordField('Password',
                            validators=[DataRequired(message='Password is required'),
                                      Length(min=6, max=128, message='Password must be at least 6 characters')])
    confirm_password = PasswordField('Confirm Password',
                                    validators=[DataRequired(message='Please confirm your password'),
                                              EqualTo('password', message='Passwords must match')])
    role = SelectField('Role',
                      choices=[('patient', 'Patient'), ('doctor', 'Doctor')],
                      validators=[DataRequired(message='Please select a role')])
    age = IntegerField('Age',
                      validators=[Optional()])
    gender = SelectField('Gender',
                        choices=[('', 'Select Gender'), ('male', 'Male'), ('female', 'Female'), ('other', 'Other')],
                        validators=[Optional()])
    phone = StringField('Phone Number',
                       validators=[Optional(), Length(max=20)])
    address = TextAreaField('Address',
                          validators=[Optional(), Length(max=500)])
    district = StringField('District',
                          validators=[Optional(), Length(max=100)])
    
    def validate_username(self, field):
        """
        Check if username already exists.
        """
        if User.query.filter_by(username=field.data).first():
            raise ValidationError('Username already exists')
    
    def validate_email(self, field):
        """
        Check if email already exists.
        """
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Email already registered')
