#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template
import joblib


# In[2]:


app = Flask(__name__)


# In[3]:



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        Purchases = request.form.get("Purchases")
        SuppCard = request.form.get("SuppCard")

        print(Purchases)
        print(SuppCard)
        
        model1 = joblib.load("CCU_Reg")
        pred1 = model1.predict([[float(Purchases), float(SuppCard)]])
        print(pred1)
        out1 = "Predicted CCU base on Linear Regression Model is: " + str(pred1)

        model2 = joblib.load("CCU_DT")
        pred2 = model2.predict([[float(Purchases), float(SuppCard)]])
        print(pred2)
        out2 = "Predicted CCU base on Decision Tree Model is: " + str(pred2)

        model3 = joblib.load("CCU_GBC")
        pred3 = model3.predict([[float(Purchases), float(SuppCard)]])
        print(pred3)
        out3 = "Predicted CCU base on GBC is: " + str(pred3)

        model4 = joblib.load("CCU_NN")
        pred4 = model4.predict([[float(Purchases), float(SuppCard)]])
        print(pred4)
        out4 = "Predicted CCU base on Neural Network Model is: " + str(pred4)

        model5 = joblib.load("CCU_RF")
        pred5 = model5.predict([[float(Purchases), float(SuppCard)]])
        print(pred5)
        out5 = "Predicted CCU base on Random Forest Model is: " + str(pred5)
        
        
        return(render_template("index.html", result1=out1, result2=out2, result3=out3, result4=out4, result5=out5))
        #return(render_template("index.html", result1=out1, result2=out2, result3=out3))
        
    else:
        return(render_template("index.html", result1="Ha", result2="Ha", result3="Ha"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




