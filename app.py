from flask import Flask, request, render_template
import joblib


# In[2]:


app = Flask(__name__)


# In[3]:



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        Income = request.form.get("Income")
        Age = request.form.get("Age")
        Loan = request.form.get("Loan")

        print(Income)
        print(Age)
        print(Loan)
        
        model1 = joblib.load("CCD_Reg")
        pred1 = model1.predict([[float(Income), float(Age), float(Loan)]])
        print(pred1)
        out1 = "Predicted credit card default based on Linear Regression Model is: " + str(pred1)

        model2 = joblib.load("CCD_DT")
        pred2 = model2.predict([[float(Income), float(Age), float(Loan)]])
        print(pred2)
        out2 = "Predicted credit card default based on Decision Tree Model is: " + str(pred2)

        model3 = joblib.load("CCD_GBC")
        pred3 = model3.predict([[float(Income), float(Age), float(Loan)]])
        print(pred3)
        out3 = "Predicted credit card default based on GBC is: " + str(pred3)

        model4 = joblib.load("CCD_NN")
        pred4 = model4.predict([[float(Income), float(Age), float(Loan)]])
        print(pred4)
        out4 = "Predicted credit card default based on Neural Network Model is: " + str(pred4)

        model5 = joblib.load("CCD_RF")
        pred5 = model5.predict([[float(Income), float(Age), float(Loan)]])
        print(pred5)
        out5 = "Predicted credit card default based on Random Forest Model is: " + str(pred5)
        
        
        return(render_template("index.html", result1=out1, result2=out2, result3=out3, result4=out4, result5=out5))
        #return(render_template("index.html", result1=out1, result2=out2, result3=out3))
        
    else:
        return(render_template("index.html", result1="Ha", result2="Ha", result3="Ha"))


# In[ ]:


if __name__ == "__main__":
    app.run()
