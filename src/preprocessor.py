# preprocessing and transformations
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


numeric_features = [
    "Log CO2 Emissions(g/km)",
    "Log Engine Size(L)",
    "Log Fuel Consumption(Hwy (L/100 km))",
    "Log Fuel Consumption(City (L/100 km))",
]
binary_features = [
    "AllOrFourWheelDrive",
    "isCVT"
]
categorical_features = [
    "Cylinders",
    "NumberOfGears",
    "Make", 
    "Vehicle Class",
    "Transmission",
    "Fuel Type",
]
target = "Smog Rating"


# ' Make the prepcocessor.
# '
# ' @return Return a preprocessor integrating transformers and encoders.
def preprocesser():
    p = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(drop="if_binary"), binary_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features))
    return p
