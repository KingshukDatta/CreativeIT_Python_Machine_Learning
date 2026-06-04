from flask import Flask, render_template, request
import pickle
import pandas as pd
import re  # <--- FIX 1: Imported the Regular Expression library!

app = Flask(__name__, template_folder="templates") 

# Load the models
lr_model = pickle.load(open("logistic_regression_model.pkl", "rb"))
tfidf_model = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# <--- FIX 2: Moved this helper function outside the routes
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)   # Remove HTML
    text = text.lower()                  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
    return text

@app.route("/") 
def pred():
    return render_template("index.html") 

# <--- FIX 3: Route is exactly on top of the function it controls
@app.route("/predict", methods=["GET","POST"])
def predict():
    # 1. Get the text data from the form
    raw_text = request.form.get('review_text')

    # Safeguard: If no text was submitted, just reload the home page
    if not raw_text:
        return render_template('index.html')

    # 2. Clean, transform, and predict
    cleaned_text = clean_text(raw_text)
    print(f"DEBUG - Cleaned Text: '{cleaned_text}'") # See what the model actually reads
    
    vectorized_text = tfidf_model.transform([cleaned_text])
    prediction = lr_model.predict(vectorized_text)[0]
    print(f"DEBUG - Raw Prediction: '{prediction}'") # See what the model actually outputs
    
    # Check if the model is outputting strings OR numbers
    if str(prediction).lower() == 'positive' or str(prediction) == '1':
        css_color = "positive"
        final_text = "Positive"
    else:
        css_color = "negative"
        final_text = "Negative"

    return render_template('index.html', final_result=final_text, result_color=css_color)

if __name__ == "__main__":
    app.run(debug=True)