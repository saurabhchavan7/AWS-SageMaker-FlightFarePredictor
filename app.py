import pickle

import os
import warnings

import joblib
import numpy as np
import pandas as pd

import xgboost as xgb

import streamlit as st

import sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    PowerTransformer,
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler
    
    
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance

from feature_engine.encoding import (
    RareLabelEncoder,
    MeanEncoder,
    CountFrequencyEncoder
    
)
#import matplotlib.pyplot as plt
import warnings



# Convenience Functions

sklearn.set_config(transform_output="pandas")


# Preprocessing Operations


# airline

airline_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    
])

# date_of_journey

feature_to_extract = ["month", "week", "day_of_week","day_of_year"]

doj_transformer = Pipeline(steps=[
     ("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True)),
     ("scaler", MinMaxScaler())
])


# source & destination

location_pipe1 = Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="other", n_categories=2)),
    ("encoder", MeanEncoder()),
    ("scaler", PowerTransformer())
])

def is_north(X):
    columns = X.columns.to_list()
    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    return (
        X
        .assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

location_transformer = FeatureUnion(transformer_list=[
    ("part1", location_pipe1),
    ("part2", FunctionTransformer(func=is_north))
])

# dep_time and arrival_time

time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
    ("scale", MinMaxScaler())
])

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
    columns = X.columns.to_list()
    X_temp = X.assign(**{
        col: pd.to_datetime(X.loc[:, col]).dt.hour
        for col in columns
    })
    
    return (
        X_temp
        .assign(**{
            f"{col}_part_of_day": np.select(
                [X_temp.loc[:, col].between(morning, noon, inclusive="left"), # inclusive = left means if value is greater than equal to left value less than right value if this is the case will call it morning  
                X_temp.loc[:, col].between(noon, eve, inclusive="left"),
                X_temp.loc[:, col].between(eve, night, inclusive="left")],
                ["morning", "afternoon","evening"],
                default="night"
            )
            for col in columns
            
        })
        .drop(columns=columns)
        
    )

time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()), # time would be out of range to bring it down will do scaling 
    ("scaler", MinMaxScaler())
])

time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2)
])

# duration


class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    # Implement the fit method
    def fit(self, X, y=None):
        # If the user has not specified the variables, identify all numeric variables in the dataset
        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()
        
        # Calculate the reference values (percentiles) for each variable
        self.reference_values_ = {
            col: (
                X
                .loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }
        
        return self

    # Implement the transform method
    def transform(self, X):
        objects = []
        for col in self.variables:
            # Create column names based on the percentiles
            columns = [f"{col}_rbf_{int(percentiles*100)}" for percentiles in self.percentiles]
            # Calculate the RBF kernel similarity and store it in a DataFrame
            obj = pd.DataFrame(
                data=rbf_kernel(X.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
        # Concatenate the transformed columns horizontally
        return pd.concat(objects, axis=1)


def duration_category(X, short=180, med=400):
    """
    Categorize flight duration into 'short', 'medium', and 'long'.

    Parameters:
    X (DataFrame): Input dataframe with 'duration' column.
    short (int): Upper limit for 'short' duration category.
    med (int): Upper limit for 'medium' duration category.

    Returns:
    DataFrame: Dataframe with 'duration' categorized and the original 'duration' column dropped.
    """
    return (
        X
        .assign(
            duration_cat=np.select(
                [X.duration.lt(short),  # If duration is less than 'short', categorize as 'short'
                 X.duration.between(short, med, inclusive="left")],  # If duration is between 'short' and 'med', categorize as 'medium'
                ["short", "medium"],  # Categories
                default="long"  # Default category if none of the conditions are met
            )
        )
        .drop(columns="duration")  # Drop the original 'duration' column
    )

def is_over(X, value=1000):
    """
    Create a binary column indicating if duration is over a specified value.

    Parameters:
    X (DataFrame): Input dataframe with 'duration' column.
    value (int): Threshold value to compare the 'duration' against.

    Returns:
    DataFrame: Dataframe with a new binary column 'duration_over_<value>' and the original 'duration' column dropped.
    """
    return (
        X
        .assign(**{
            f"duration_over_{value}": X.duration.ge(value).astype(int)  # Create new column indicating if 'duration' is greater than or equal to 'value'
        })
        .drop(columns="duration")  # Drop the original 'duration' column
    )


duration_pipe1 = Pipeline(steps=[
    ("rbf", RBFPercentileSimilarity()),  # Apply RBF Percentile Similarity transformation
    ("scaler", PowerTransformer())       # Apply Power Transformation
])

duration_pipe2 = Pipeline(steps=[
    ("cat", FunctionTransformer(func=duration_category)),  # Categorize duration
    ("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))  # Encode categories as ordinal
])

duration_union = FeatureUnion(transformer_list=[
    ("part1", duration_pipe1),  # RBF Percentile Similarity + Power Transformation
    ("part2", duration_pipe2),  # Duration Categorization + Ordinal Encoding
    ("part3", FunctionTransformer(func=is_over)),  # Create binary column for duration over a specified value
    ("part4", StandardScaler())  # Standardize the features
])

# Define a pipeline for overall duration transformation
duration_transformer = Pipeline(steps=[
    ("outliers", Winsorizer(capping_method="iqr", fold=1.5)),  # Handles outliers using Winsorization
    ("imputer", SimpleImputer(strategy="median")),  # Imputes missing values using median strategy
    ("union", duration_union)  # Combines all transformations using FeatureUnion
])

# total_stops

def is_direct(X):
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))


total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("", FunctionTransformer(func=is_direct))
])

# additional_info

info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


def have_info(X):
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
    ("part1", info_pipe1),
    ("part2", FunctionTransformer(func=have_info))
])


info_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("union", info_union)
])

# Column Transformer

column_transformer = ColumnTransformer(transformers=[
    ("air", airline_transformer, ["airline"]),
    ("doj", doj_transformer, ["date_of_journey"]),
    ("location", location_transformer, ["source", "destination"]), # on source and destination column it will perform the location_transformer transformation 
    ("time", time_transformer, ["dep_time", "arrival_time"]),
    ("dur", duration_transformer, ["duration"]),
    ("stops", total_stops_transformer, ["total_stops"]),
    ("info", info_transformer, ["additional_info"])
], remainder="passthrough")

# Feature Selector

estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
    estimator=estimator,
    scoring="r2",
    threshold=0.1
) 

# preprocessor

preprocessor = Pipeline(steps=[
    ("ct", column_transformer),
    ("selector", selector)
])

# read the training data

dir_path = r"D:\SAURABH\spring 2024\ML_Projects\AWS SageMaker FlightFarePredictor\data"
train = pd.read_csv(os.path.join(dir_path, "train_set.csv"))
X_train = train.drop(columns="price") 
y_train = train.price.copy()

# fit and save the preprocessor

preprocessor.fit(X_train, y_train)
joblib.dump(preprocessor, "preprocessor.joblib") 


# Web Application

st.set_page_config(
    page_title="Flights Price Prediction",
    page_icon="🛫",
    layout="centered" # so application can use entire width of the webpage  
)

st.title("AWS-SageMaker-Flight Fare Predictor")

# now the initial 9 columns will take that as input fromuser

# user inputs

airline = st.selectbox(
    "Airline:",
    options=X_train.airline.unique()
)

doj = st.date_input("Data of Journey:")

source = st.selectbox(
    "Source:",
    options=X_train.source.unique()
)

destination = st.selectbox(
    "Destination:",
    options=X_train.destination.unique()
)

dep_time = st.time_input("Departure Time:")

arrival_time = st.time_input("Arrival Time:")

duration = st.number_input(
    "Duration (mins):",
    step=1
)

total_stops = st.number_input(
    "Total Stops:",
    step=1,
    min_value=0
)

additional_info = st.selectbox(
    "Additional Info:",
    options=X_train.additional_info.unique()
)


# to transform the data we have to convert it into pandas dataframe

x_new = pd.DataFrame(dict(
    airline=[airline],
    date_of_journey=[doj],
    source=[source],
    destination=[destination],
    dep_time=[dep_time],
    arrival_time=[arrival_time],
    duration=[duration],
    total_stops=[total_stops],
    additional_info=[additional_info]
)).astype({
    col: "str"
    for col in ["date_of_journey", "dep_time", "arrival_time"]
})


# now this we have to transform but there is a catch the date input and time input for doj, dep_time of streamlit was date time object
# but pandas is running its own function pd.todatetime
# so pandas to date time function will not work on streamlit date time object 
# so this 3 will have to convert the dtype as a string so when we pass it in our transformer so pandas would easily convert from sting type to its own sate type 

# now what we have to do of x_new?
# we have to preprocess this and we have our preprocessor saved so will load that


if st.button("Predict"):
    saved_preprocessor = joblib.load("preprocessor.joblib")
    x_new_pre = saved_preprocessor.transform(x_new) # this will perform transformation on user input and will save it in x_new_pre

    # now we need to predict this for this we need model which we have saved so will load it

    with open("xgboost-model", "rb") as f:
        model = pickle.load(f)
    x_new_xgb = xgb.DMatrix(x_new_pre)
    pred = model.predict(x_new_xgb)[0] # this prediction will return a array on 1 value but we want value alone so will take it out using [0]

    # whatever we pass inside info ist shows in blue box

    st.info(f"The Predicted price is {pred:,.0f} INR")