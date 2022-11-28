import pandas as pd
import ast

def remove_rows_with_missing_values(df):
    df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'Description'], inplace=True)
    return df


def clean_description_strings(x: str):
    try:
        cleaned_x = ast.literal_eval(x)
        cleaned_x.remove("About this space")
        cleaned_x = "".join(cleaned_x)
        return cleaned_x
    except Exception as e:
        return x

def combine_description_strings(df):
    df["Description"] = df["Description"].apply(clean_description_strings)
    df['Description'].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex=True, inplace=True)
    return df
     

def set_default_feature_values(df):
    column_list = ["guests", "beds", "bathrooms", "bedrooms"]
    df[column_list] = df[column_list].fillna(1)
    return df

def clean_tabular_data(df):
    df = remove_rows_with_missing_values(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df

def load_airbnb(df, label_col):
    labels = df[label_col]
    features = df.drop(label_col, axis=1)
    return (features, labels)

if __name__ == "__main__":
    df_listing = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv')
    df_listing.drop('Unnamed: 19', axis=1, inplace=True)
    df_clean_listing = clean_tabular_data(df_listing)
    df_clean_listing.to_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')    
