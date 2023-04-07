# preprocessing and transformations
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

numeric_feature = [
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


def preprocesser():
    p = make_column_transformer(
        (StandardScaler(), numeric_feature),
        (OneHotEncoder(drop="if_binary"), binary_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features))
    return p

# X_train_transformed = preprocessor.fit_transform(X_train)
# X_test_transformed = preprocessor.transform(X_test)

# X_train_df = pd.DataFrame(X_train_transformed.toarray())
# X_test_df = pd.DataFrame(X_test_transformed.toarray())