# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories files as pandas dataframes, merge them together
    and return the merged dataframe
    
    Pars:
    - messages_filepath: filepath of csv file with messages
    - categories_filepath: filepath of csv file with categories
    
    Outs:
    - df: pandas dataframe with messages and categories
    
    """
    
    # Load messages and categories files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on= "id")
    
    return df

def clean_data(df):
        
    # Split `categories` into separate category columns.
    df_categories = df.categories.str.split(pat = ";", expand=True)

    # Use the first row of categories dataframe to create column names for the categories data
    row = df_categories.iloc[0]
    category_colnames = [st.split("-")[0] for st in row]
    df_categories.columns = category_colnames
    
    # Convert category values to numbers 0 or 1
    for column in df_categories:
        # set each value to be the last character of the string
        df_categories[column] = df_categories[column].str.split(pat = "-", expand=True).iloc[:, 1]
    
        # convert column from string to numeric
        df_categories[column] = df_categories[column].astype(int)
        
        # make sure only 0 or 1 are present
        df_categories[column] = df_categories[column].apply(lambda x: 0 if x<1 else 1)
        
    # Replace `categories` column in `df` with new category columns.
    df.drop(columns = ["categories"], inplace=True)
    df = pd.concat([df, df_categories], axis = 1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///disaster_df.db')
    df.to_sql('disaster_df', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()         