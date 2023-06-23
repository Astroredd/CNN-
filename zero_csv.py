import pandas as pd
import os 
import numpy as np

columns = [['Class1.1'], ['Class1.2'], ['Class1.3'], ['Class2.1'], ['Class2.2'], ['Class3.1'],['Class3.2'], ['Class4.1'], ['Class4.2'], ['Class5.1'], ['Class5.2'], ['Class5.3'],['Class5.4'], ['Class6.1'],['Class6.2'],['Class7.1'], ['Class7.2'], ['Class7.3'],['Class8.1'], ['Class8.2'], ['Class8.3'], ['Class8.4'], ['Class8.5'], ['Class8.6'], ['Class8.7'], ['Class9.1'], ['Class9.2'], ['Class9.3'],['Class10.1'],['Class10.2'], ['Class10.3'], ['Class11.1'],[ 'Class11.2'], ['Class11.3'], ['Class11.4'], ['Class11.5'], ['Class11.6']]


lista_de_archivos = os.listdir('directory') 
df = pd.DataFrame()

df['GalaxyID'] = lista_de_archivos
for i in columns:
    df[i] = np.NaN

df = df.fillna(0)
def Clear_id(df):
    return df.replace(".png", "").split("/")[0]

df["GalaxyID"] = df["GalaxyID"].apply(Clear_id)
print(df.head())

df.to_csv("Zeros.csv",index=False)
