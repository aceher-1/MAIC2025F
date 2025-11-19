import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

BASIC_FEATURES = ["age", "bmi", "sex", "smoker", "region"]
ADVANCED_FEATURES = BASIC_FEATURES + [
    "children",
    "AnyTransplants",
    "AnyChronicDiseases",
    "HistoryOfCancerInFamily",
    "KnownAllergies",
    "NumberOfMajorSurgeries",
]

BASIC_CAT = ["sex", "smoker", "region"]
ADVANCED_CAT = BASIC_CAT + [
    "AnyTransplants",
    "AnyChronicDiseases",
    "HistoryOfCancerInFamily",
    "KnownAllergies",
]

@st.cache_resource
def load_basic_model():
    m = CatBoostRegressor()
    m.load_model("premium_catboost_basic.cbm")
    return m

@st.cache_resource
def load_advanced_model():
    m = CatBoostRegressor()
    m.load_model("premium_catboost_advanced.cbm")
    return m

basic_model    = load_basic_model()
advanced_model = load_advanced_model()

st.title("Premium Estimator")

st.write("Choose a simple quick estimate or include more medical history for a more detailed estimate.")

model_choice = st.radio(
    "Model type",
    ("Quick (basic features only)", "Advanced (includes medical history)"),
    index=0,
)

# ---------- basic inputs ----------
age = st.number_input("Age", min_value=18, max_value=100, value=35)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5, step=0.1)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ---------- advanced inputs (optional, with safe defaults) ----------
with st.expander("Advanced medical info (optional)"):
    st.write("If you're not sure, you can leave these as-is.")

    children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

    AnyTransplants = st.selectbox("Any transplants?", ["No", "Yes"], index=0)
    AnyChronicDiseases = st.selectbox("Any chronic diseases?", ["No", "Yes"], index=0)
    HistoryOfCancerInFamily = st.selectbox(
        "Family history of cancer?", ["No", "Yes"], index=0
    )
    KnownAllergies = st.selectbox("Known allergies?", ["No", "Yes"], index=0)

    NumberOfMajorSurgeries = st.number_input(
        "Number of major surgeries", min_value=0, max_value=10, value=0
    )

# ---------- build input rows ----------

basic_row = {
    "age": age,
    "bmi": bmi,
    "sex": sex,
    "smoker": smoker,
    "region": region,
}

advanced_row = {
    "age": age,
    "bmi": bmi,
    "sex": sex,
    "smoker": smoker,
    "region": region,
    "children": children,
    "AnyTransplants": AnyTransplants,
    "AnyChronicDiseases": AnyChronicDiseases,
    "HistoryOfCancerInFamily": HistoryOfCancerInFamily,
    "KnownAllergies": KnownAllergies,
    "NumberOfMajorSurgeries": NumberOfMajorSurgeries,
}

basic_df = pd.DataFrame([basic_row])[BASIC_FEATURES].copy()
advanced_df = pd.DataFrame([advanced_row])[ADVANCED_FEATURES].copy()

basic_df[BASIC_CAT] = basic_df[BASIC_CAT].astype(str)
advanced_df[ADVANCED_CAT] = advanced_df[ADVANCED_CAT].astype(str)

st.write("Input (basic features):")
st.dataframe(basic_df)

if model_choice.startswith("Advanced"):
    st.write("Input (advanced features):")
    st.dataframe(advanced_df)

# ---------- prediction ----------
if st.button("Estimate Premium"):
    if model_choice.startswith("Quick"):
        pred = basic_model.predict(basic_df)[0]
        st.subheader(f"Estimated premium (basic model): {pred:,.2f}")
    else:
        pred_basic = basic_model.predict(basic_df)[0]
        pred_adv   = advanced_model.predict(advanced_df)[0]

        st.subheader(f"Estimated premium (advanced model): {pred_adv:,.2f}")
        st.caption(f"(Quick/basic estimate for comparison: {pred_basic:,.2f})")
