import pandas as pd
import numpy as np


def prepare_data(path):
    ds_name = path.name
    if ds_name == 'adult':
        df = pd.read_csv(path/'adult.csv')

        dep_var = '>=50k'
        num_vars = [
            'age', 'fnlwgt', 'education-num', 
            'hours-per-week', 'capital-gain', 'capital-loss']
        cat_vars = [
            'workclass', 'education', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'native-country']

    elif ds_name == 'forest':
        df = pd.read_csv('../data/forest/covtype.data', header=None)
        soil_types = ['2702','2703','2704','2705','2706','2717','3501','3502','4201','4703','4704','4744',
        '4758','5101','5151','6101','6102','6731','7101','7102','7103','7201','7202','7700',
        '7701','7702','7709','7710','7745','7746','7755','7756','7757','7790','8703','8707',
        '8708','8771','8772','8776']

        df.columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_Rawah', 'Wilderness_Area_Neota', 
                    'Wilderness_Area_Comanche', 'Wilderness_Area_Cache'] + [f'Soil_Type_{i}' for i in soil_types] + ['Cover_Type']

        col_ind = df.columns[df.columns.str.startswith('Wilderness_Area')]
        df['Wilderness_Area'] = df.loc[:, col_ind].idxmax(1).drop(columns=col_ind)
        df.drop(columns=col_ind, inplace=True)

        col_ind = df.columns[df.columns.str.startswith('Soil_Type')]
        df['Soil_Type'] = df.loc[:, col_ind].idxmax(1)
        df.drop(columns=col_ind, inplace=True)
        df['Soil_Type'] = df['Soil_Type'].str.split('_').str[2]
        df['Soil_Type_1'] = df['Soil_Type'].str[0]
        df['Soil_Type_2'] = df['Soil_Type'].str[1]
        df.drop(columns='Soil_Type', inplace=True)

        dep_var = 'Cover_Type'
        num_vars = list(df.columns[:10])
        cat_vars = ['Wilderness_Area', 'Soil_Type_1', 'Soil_Type_2']

    return df, dep_var, num_vars, cat_vars