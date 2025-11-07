# Authentication Setup

## Overview
Simple authentication system with sign in/sign up functionality using Flask-Login and SQLite database.

## Features
- User registration (sign up)
- User login (sign in)
- Session management
- Protected routes (requires login to access main app)
- SQLite database for user storage
- Password hashing for security

## Installation

Install the required package:
```bash
pip install flask-login
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Navigate to `http://localhost:5001`

3. You'll be redirected to the login page

4. Create a new account by clicking "Sign Up"

5. After registration, sign in with your credentials

6. You'll be redirected to the main lung cancer classification app

## Routes

- `/login` - Sign in page
- `/signup` - Sign up page
- `/logout` - Logout (requires authentication)
- `/` - Main app (requires authentication)
- `/predict` - Prediction endpoint (requires authentication)

## Database

The app uses SQLite database (`users.db`) which is automatically created on first run. The database stores:
- User ID
- Username (unique)
- Email (unique)
- Hashed password

## Security Notes

- Passwords are hashed using Werkzeug's security functions
- Change the `SECRET_KEY` in `app.py` for production use
- The database file is excluded from git via `.gitignore`
