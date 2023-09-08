import streamlit as st
import pickle
import numpy as np

ml = pickle.load(open('ml.pkl', 'rb'))

def perdict_forest(Oxygen, Temperature, Humidity):
    input = np.array([[Oxygen, Temperature, Humidity]]).astype(np.float64)
    prediction = ml.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    print(type(pred))
    return float(pred)

def main():
    st.title("Forest Fire Prediction")
    html_temp = """
    <div style = "background-color:#025246 ;padding:10px">
    <h2 style = "color:white;text-align:center;">Forest Fire Prediction ML Appliaction </h2>
    </div> 
    """

    st.markdown(html_temp, unsafe_allow_html = True)

    Oxygen = st.text_input("Oxygen", "Type Here")
    Temperature = st.text_input("Temperature", "Type Here")
    Humidity = st.text_input("Humidity", "Type Here")
    
    safe_html = """
    <div style = "background-color:#F08080;padding:10px">
    <h2 style = "color:white;text-align:center;"> Your forest is safe</h2>
    </div>
    """

    danger_html = """
    <div style = "background-color:#F08080;padding:10px">
    <h2 style = "color:black;text-align:center;"> Your forest is in danger</h2>
    </div>
    """
    
    if st.button("Predict"):
        output = perdict_forest(Oxygen, Temperature, Humidity)
        st.success("The probablibility of fire taking place is {}".format(output))

        if output > 0.5:
            st.markdown(danger_html, unsafe_allow_html = True)
        else:
            st.markdown(safe_html, unsafe_allow_html = True)

if __name__ == "__main__":
    main()

