import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.preprocessor import *

encoding_name_list = [
'standardscaler__Log CO2 Emissions(g/km)',
'standardscaler__Log Engine Size(L)',
'standardscaler__Log Fuel Consumption(Hwy (L/100 km))',
'standardscaler__Log Fuel Consumption(City (L/100 km))',
'standardscaler__Log CO2 Emissions(g/km)*Log Engine Size(L)',
'standardscaler__Log CO2 Emissions(g/km)*Log Fuel Consumption(Hwy (L/100 km))',
'standardscaler__Log CO2 Emissions(g/km)*Log Fuel Consumption(City (L/100 km))',
'standardscaler__Log Engine Size(L)*Log CO2 Emissions(g/km)',
'standardscaler__Log Engine Size(L)*Log Fuel Consumption(Hwy (L/100 km))',
'standardscaler__Log Engine Size(L)*Log Fuel Consumption(City (L/100 km))',
'standardscaler__Log Fuel Consumption(Hwy (L/100 km))*Log CO2 Emissions(g/km)',
'standardscaler__Log Fuel Consumption(Hwy (L/100 km))*Log Engine Size(L)',
'standardscaler__Log Fuel Consumption(Hwy (L/100 km))*Log Fuel Consumption(City (L/100 km))',
'standardscaler__Log Fuel Consumption(City (L/100 km))*Log CO2 Emissions(g/km)',
'standardscaler__Log Fuel Consumption(City (L/100 km))*Log Engine Size(L)',
'standardscaler__Log Fuel Consumption(City (L/100 km))*Log Fuel Consumption(Hwy (L/100 km))',
'onehotencoder-1__AllOrFourWheelDrive_True',
'onehotencoder-1__isCVT_True',
'onehotencoder-2__Cylinders_equal to 6',
'onehotencoder-2__Cylinders_larger/equal to 8',
'onehotencoder-2__Cylinders_smaller/equal to 5',
'onehotencoder-2__NumberOfGears_10.0',
'onehotencoder-2__NumberOfGears_7.0',
'onehotencoder-2__NumberOfGears_7.959025470653378',
'onehotencoder-2__NumberOfGears_8.0',
'onehotencoder-2__NumberOfGears_9.0',
'onehotencoder-2__NumberOfGears_smaller/equal to 6',
'onehotencoder-2__Make_Acura',
'onehotencoder-2__Make_Alfa Romeo',
'onehotencoder-2__Make_Aston Martin',
'onehotencoder-2__Make_Audi',
'onehotencoder-2__Make_BMW',
'onehotencoder-2__Make_Bentley',
'onehotencoder-2__Make_Bugatti',
'onehotencoder-2__Make_Buick',
'onehotencoder-2__Make_Cadillac',
'onehotencoder-2__Make_Chevrolet',
'onehotencoder-2__Make_Chrysler',
'onehotencoder-2__Make_Dodge',
'onehotencoder-2__Make_FIAT',
'onehotencoder-2__Make_Ford',
'onehotencoder-2__Make_GMC',
'onehotencoder-2__Make_Genesis',
'onehotencoder-2__Make_Honda',
'onehotencoder-2__Make_Hyundai',
'onehotencoder-2__Make_Infiniti',
'onehotencoder-2__Make_Jaguar',
'onehotencoder-2__Make_Jeep',
'onehotencoder-2__Make_Kia',
'onehotencoder-2__Make_Lamborghini',
'onehotencoder-2__Make_Land Rover',
'onehotencoder-2__Make_Lexus',
'onehotencoder-2__Make_Lincoln',
'onehotencoder-2__Make_MINI',
'onehotencoder-2__Make_Maserati',
'onehotencoder-2__Make_Mazda',
'onehotencoder-2__Make_Mercedes-Benz',
'onehotencoder-2__Make_Mitsubishi',
'onehotencoder-2__Make_Nissan',
'onehotencoder-2__Make_Porsche',
'onehotencoder-2__Make_Ram',
'onehotencoder-2__Make_Rolls-Royce',
'onehotencoder-2__Make_Subaru',
'onehotencoder-2__Make_Toyota',
'onehotencoder-2__Make_Volkswagen',
'onehotencoder-2__Make_Volvo',
'onehotencoder-2__Vehicle Class_Large',
'onehotencoder-2__Vehicle Class_Mid-size',
'onehotencoder-2__Vehicle Class_Pickup truck: Standard',
'onehotencoder-2__Vehicle Class_SUV: Small',
'onehotencoder-2__Vehicle Class_SUV: Standard',
'onehotencoder-2__Vehicle Class_Small',
'onehotencoder-2__Vehicle Class_Smallest',
'onehotencoder-2__Transmission_A',
'onehotencoder-2__Transmission_AM',
'onehotencoder-2__Transmission_AS',
'onehotencoder-2__Transmission_AV',
'onehotencoder-2__Transmission_M',
'onehotencoder-2__Fuel Type_H',
'onehotencoder-2__Fuel Type_L']


def make_interaction_set(X_train, X_test):
  X_train = pd.DataFrame(X_train)
  X_test = pd.DataFrame(X_test)
  dataset = X_train
  output_intr_train = pd.DataFrame()
  output_intr_test = pd.DataFrame()
  for col in dataset:
    output_intr_train[col] = dataset[col]
    output_intr_test[col] = X_test[col]
  numeric_features_intr = []
  for nm in numeric_features:
    numeric_features_intr.append(nm)
  for col1 in dataset:
    for col2 in output_intr_train:
      if (col1 in numeric_features) and (col2 in numeric_features) and (col1 != col2):
        output_intr_train[col1+'*'+col2] = X_train[col1]*X_train[col2]
        output_intr_test[col1+'*'+col2] = X_test[col1]*X_test[col2]
        name = col1+'*'+col2
        numeric_features_intr.append(name)

  preprocessor_intr = make_column_transformer(
      (StandardScaler(), numeric_features_intr),
      (OneHotEncoder(drop="if_binary"), binary_features),
      (OneHotEncoder(handle_unknown="ignore"), categorical_features),
  )
  out_list = {}
  output_df_train = pd.DataFrame(preprocessor_intr.fit_transform(output_intr_train).toarray())
  out_list['train'] = output_df_train
  output_df_test = pd.DataFrame(preprocessor_intr.transform(output_intr_test).toarray())
  out_list['test'] = output_df_test
  out_list['input_columns'] = output_intr_train.columns

  return out_list