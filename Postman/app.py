from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("D:\CreativeIT Class work\Postman\ titan_model.lib")

@app.route("/predict", methods= ["POST"])
def pred():
    p_class = float(request.form.get("pclass"))	
    gender= float(request.form.get("sex"))	 
    Age= float(request.form.get("age"))		
    ss= float(request.form.get("sibsp"))		
    pc= float(request.form.get("parch"))		
    rent= float(request.form.get("fare"))		
    cate= float(request.form.get("who"))		
    adult= float(request.form.get("adult_male"))		
    town= float(request.form.get("embark_town"))		
    Alone= float(request.form.get("alone"))	

    data = [p_class,gender,Age,ss,pc,rent,cate,adult,town,Alone]
    predi = model.predict([data])
    return jsonify({"predicted_class": int(predi[0])})


if __name__=="__main__":
    app.run(debug=True)