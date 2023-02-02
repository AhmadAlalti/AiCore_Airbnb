import pandas as pd
import ast

def remove_rows_with_missing_values(df):
    df.dropna(inplace=True)
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

def load_airbnb(df, label_col):
    labels = df[label_col]
    features = df.drop(label_col, axis=1)
    return (features, labels)

def clean_amenities_strings(x: str):
    try:
        cleaned_x = ast.literal_eval(x)
        cleaned_x.remove('What this place offers')
        cleaned_x = " ".join(cleaned_x)
        return cleaned_x
    except Exception as e:
        return x

def combine_amenities_strings(df):
    df["Amenities"] = df["Amenities"].apply(clean_amenities_strings)
    df['Amenities'].replace([r"\\n", "\n", r"\'"], [" "," "," "], regex=True, inplace=True)
    return df

if __name__ == "__main__":
    df_listing = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv', index_col='ID')
    df_listing.drop('Unnamed: 19', axis=1, inplace=True)
    set_default_feature_values(df_listing)
    remove_rows_with_missing_values(df_listing)
    combine_description_strings(df_listing)
    combine_amenities_strings(df_listing)
    df_listing.drop('d577bc30-2222-4bef-a35e-a9825642aec4', inplace=True)
    df_listing.drop('4c917b3c-d693-4ee4-a321-f5babc728dc9', inplace=True)
    df_listing['guests'] = df_listing['guests'].astype('int64')
    df_listing['bedrooms'] = df_listing['bedrooms'].astype('int64')
    df_listing['Category'] = df_listing['Category'].astype('category')
    df_listing.to_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
