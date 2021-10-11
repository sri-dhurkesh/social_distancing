import pyrebase
config = {
  "apiKey": "AIzaSyD9LcPsV7StilRfvnHMtHAWGSiPmaYX7Ic",
  "authDomain": "social-distancing-4791d.firebaseapp.com",
  "databaseURL": "https://social-distancing-4791d-default-rtdb.firebaseio.com",
  "projectId": "social-distancing-4791d",
  "storageBucket": "social-distancing-4791d.appspot.com",
}
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

# Log the user in
user = auth.sign_in_with_email_and_password("svdhurkesh68@gmail.com", "Santeasdf@123")

# Log the user in anonymously
user = auth.sign_in_anonymous()

# Get a reference to the database service
db = firebase.database()

# data to save
data = {
    "name": "Mortimer 'Morty' Smith"
}

# Pass the user's idToken to the push method
results = db.child("users").push(data, user['idToken'])
