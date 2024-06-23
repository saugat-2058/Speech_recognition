from flask import Flask, render_template, jsonify, request, session, redirect
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import not_
from threading import Thread
import os 
import re
import bcrypt
from final import SpeechRecognizer

# Create an instance of Flask
app = Flask(__name__)
app.secret_key = "your_secret_key_here"
# Configure Flask app to connect to MySQL database
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "mysql+mysqlconnector://root:@localhost/asr_users"
)

# # Suppress deprecation warning
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


db = SQLAlchemy(app)
# print(db)


class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return "<User %r>" % self.username


# Configure static folder
app.static_folder = "static"


# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/users")
def get_users():
    users = Users.query.all()  # Perform a query to get all users
    user_list = [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "password": user.password,
        }
        for user in users
    ]
    return jsonify(user_list)


@app.route("/speech-recon")
def speech_recon():
    return render_template("indexs.html", user=session.get("user"))
    # return render_template('indexs.html')


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/signup-submit", methods=["POST"])
def signup_submit():
    # Extract form data
    username = request.form["username"]
    email = request.form["email"]
    confirm_password = request.form.get("confirmp")
    password = request.form["password"]

    inf_message = None

    # Basic email format validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        inf_message = "Please enter a valid email address."

    # Username validation
    elif not username.strip():
        inf_message = "Username cannot be empty."

    # Password complexity validation
    elif not re.match(r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[^a-zA-Z0-9]).{6,}$", password):
        inf_message = 'Password must contain at least<br><ul><li>one uppercase letter</li><li>one lowercase letter</li><li>one special character</li><li>one digit</li><li>and be at least 8 characters long.</li></ul>'

    # Password matching validation
    elif password != confirm_password:
        inf_message = "Passwords do not match."

    if inf_message:
        return render_template("signup.html", message=inf_message)


    
    password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    # print(username)
    # print(email)
    # print(password)
    # Create a new User object
    existing_user = Users.query.filter_by(username=username).first()
    if existing_user:
        return render_template(
            "signup.html",
            message="User with the name already exists. Please try a different username.",
        )
    else:
        existing_user = Users.query.filter_by(email=email).first()

        if existing_user:
            return render_template(
                "signup.html",
                message="User with the name or email already exists. Please try a different username or email.",
            )
        else:
            new_user = Users(username=username, email=email, password=password)

            db.session.add(new_user)

            db.session.commit()

            return render_template(
                "login.html", messages="signup Successfull now proceed to login"
            )


@app.route("/login-submit", methods=["POST"])
def login_submit():
    # Extract form data
    username = request.form["username"]
    # email = request.form['email']
    password = request.form["password"]
    existing_user = Users.query.filter_by(username=username).first()
    if existing_user:
        if bcrypt.checkpw(
            password.encode("utf-8"), existing_user.password.encode("utf-8")
        ):
            #   new_user = Users(username=username)
            #   db.session.add(new_user)
            session["logged_in"] = True
            session["user"] = username
            return render_template(
                "indexs.html", message="Login Successgull", user=username
            )
        else:
            return render_template("login.html", message="invalid credentials")
    else:
        return render_template("login.html", message="invalid credentials")
    # print(username)
    # print(email)
    # print(password)
    # Create a new User object
    # new_user = Users(username=username, password=password)

    # db.session.add(new_user)

    # db.session.commit()

    # return render_template('login.html')


# Flag to check if Tkinter window is opened
tkinter_window_opened = False

# def open_tkinter_window(user):
#     global tkinter_window_opened
#     # initialize(user)


def open_tkinter_window(user):
    global tkinter_window_opened
    tkinter_window_opened = True
    recon = SpeechRecognizer()
    window = recon.show_welcome_window(user)
    window.attributes("-topmost", True)
    # window  .mainloop()
    window.mainloop()

    tkinter_window_opened = False


@app.route("/test_sys")
def test_sys():
    if "user" in session:
        user = session.get("user")
        global tkinter_window_opened
        if not tkinter_window_opened:
            thread = Thread(target=open_tkinter_window, args=(user,))
            thread.start()
        return render_template("indexs.html", user=user)
    else:
        return render_template(
            "login.html", message="Please Login In Order To Grant Access"
        )


# Route to check if the Tkinter window is opened
@app.route("/tkinter_window_status")
def tkinter_window_status():
    global tkinter_window_opened
    return jsonify({"tkinter_window_opened": tkinter_window_opened})


@app.route("/logout")
def logout():
    session.clear()
    return render_template("login.html")


@app.route("/admin")
def admin():
    if session.get("logged_in"):
        existing_user = Users.query.filter_by(username=session.get("user")).first()
        if existing_user.role == 1:
            user = session.get("user")
            users = Users.query.filter(Users.id != existing_user.id).all()
            return render_template(
                "indexsau.html", message="Welcome", user=user, users=users
            )
        else:
            return render_template("logina.html")
    else:
        return render_template("logina.html")


@app.route("/login-submita", methods=["POST"])
def login_submita():
    # Extract form data
    username = request.form["username"]
    # email = request.form['email']
    password = request.form["password"]
    existing_user = Users.query.filter_by(username=username).first()
    if existing_user:
        if bcrypt.checkpw(
            password.encode("utf-8"), existing_user.password.encode("utf-8")
        ):
            #   new_user = Users(username=username)
            #   db.session.add(new_user)
            if existing_user.role == 1:
                session["logged_in"] = True
                session["user"] = username
                user = Users.query.filter(Users.id != existing_user.id).all()
                # print(user)
                return render_template(
                    "indexsau.html",
                    message="Login Successgull",
                    user=username,
                    users=user,
                )
            else:
                return render_template(
                    "logina.html",
                    message="You Dont Have access of the admin! Cannot login",
                )
        else:
            return render_template("logina.html", message="invalid credentials")
    else:
        return render_template("logina.html", message="invalid credentials")


@app.route("/profile")
def profile():
    if session.get("logged_in"):
        user = Users.query.filter_by(username=session.get("user")).first()
        username = user.username
        user_email = user.email
        # print(user)
        return render_template("profile.html", user=username, email=user_email)
    else:
        return render_template(
            "login.html", message="Please Login In To GO TO this page"
        )


@app.route("/delete")
def delete():
    if session.get("logged_in"):
        existing_user = Users.query.filter_by(username=session.get("user")).first()
        if existing_user.role == 1:
            user_id = request.args.get("id")
            user = session.get("user")
            existing_user = Users.query.filter_by(username=user).first()
            users = Users.query.filter(Users.id != existing_user.id).all()
            if user_id:
                # Query the database to get the user object
                user = Users.query.get(user_id)
                if user:
                    # Delete the user from the database
                    db.session.delete(user)
                    db.session.commit()
                    user = session.get("user")
                    return redirect("/admin")
                else:
                    return render_template(
                        "indexsau.html",
                        message="User Not FOund",
                        user=user,
                        users=users,
                    )
            else:
                return render_template(
                    "indexsau.html",
                    message="User not found with the current id",
                    user=user,
                    users=users,
                )
        else:
            return render_template(
                "indexs.html", message="You cannot Perform this operation Access Denied"
            )
    else:
        return render_template(
            "login.html", message="Please Login In To GO TO this page"
        )


@app.route("/change_pass", methods=["POST", "GET"])
def change_pass():
    if request.method == "POST":
        password = request.form["password"]
        confirm_password = request.form["confirmp"]
        inf_message = None

   

        # Password complexity validation
        if re.match(r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[^a-zA-Z0-9]).{6,}$", password):
            inf_message = 'Password must contain at least<br><ul><li>one uppercase letter</li><li>one lowercase letter</li><li>one special character</li><li>one digit</li><li>and be at least 8 characters long.</li></ul>'

        # Password matching validation
        elif password != confirm_password:
            inf_message = "Passwords do not match."

        if inf_message:
            return render_template("pass_change.html", message=inf_message)
        
        if password == confirm_password:
            user = Users.query.filter_by(username=session.get("user")).first()
            if bcrypt.checkpw(password.encode("utf-8"), user.password.encode("utf-8")):
                return render_template(
                    "pass_change.html",
                    message="current password and previous password mustnot be same",
                )
            else:
                password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
                user.password = password
                db.session.commit()
                return render_template(
                    "profile.html",
                    message="Password changed Successfully",
                    user=user.username,
                    email=user.email,
                )
        else:
            return render_template(
                "pass_change.html", message="confirm_password and password donot match"
            )
    else:
        if "user" in session:
            return render_template("pass_change.html")
        else:
            return render_template(
                "login.html", message="Please Login In To GO TO this page"
            )


@app.route("/change_other", methods=["POST", "GET"])
def change_other():
    if request.method == "POST":
        user = Users.query.filter_by(username=session.get("user")).first()
        username = request.form["username"]
        email = request.form["email"]
        inf_message = None

        # Basic email format validation
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            inf_message = "Please enter a valid email address."

        # Username validation
        elif not username.strip():
            inf_message = "Username cannot be empty."

        if inf_message:
            return render_template("change_others.html", message=inf_message)
        users = Users.query.filter(Users.username != username).all()
        for usersd in users:
            # Check if the entered email and username match with any existing email and username in the database
            if usersd.email == email or usersd.username == username:
                return render_template(
                    "change_others.html", message="Username or email already exists"
                )
            else:
                user.email = email
                user.username = username
                session["user"] = username
                db.session.commit()
                return render_template(
                    "profile.html",
                    message="User Info Changed Successfully",
                    user=user.username,
                    email=user.email,
                )
    else:
        if "user" in session:
            return render_template("change_others.html")
        else:
            return render_template(
                "login.html", message="Please Login In To GO TO this page"
            )


@app.route("/result", methods=["GET"])
def result():
    folder_path = str(request.args.get("folder_path"))
    # folder_path="/static/"+folder_path+''
    # folder_path = request.args.get('folder_path')
    image_paths = []  # List to store image paths
    json_paths = []  # List to store JSON file paths
    audio_paths = []

    # Assuming images have a specific extension, like .png or .jpg
    image_extensions = [".png", ".jpg", ".jpeg"]
    audio_extension = [".wav"]

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            # Check if the file is an image
            if any(filename.endswith(ext) for ext in image_extensions):
                image_paths.append(f"{file_path}")
            elif filename.endswith(".json"):
                json_paths.append(f"{file_path}")
            elif filename.endswith(".wav"):
                audio_paths.append(f"{file_path}")

    return render_template(
        "result.html",
        image_paths=image_paths,
        json_paths=json_paths,
        audio_paths=audio_paths,
    )
    # return render_template('result.html', images=images, json_files=json_files)

    # return render_template("result.html")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
