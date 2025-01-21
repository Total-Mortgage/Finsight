import os
import pandas as pd
import json
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from dotenv import load_dotenv

# loading env variables
load_dotenv()

conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};' \
               'SERVER=tmssql03.tmsdomain.com\\tmssql1;' \
               'DATABASE=Salesforce_DW;' \
               'UID=' + os.environ.get('db_user') + ';' \
               'PWD=' + os.environ.get('db_pass')
connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': conn_str})
engine = create_engine(connection_url)

class contextManager:
    def __init__(self, context_folder):
        self.root = context_folder
        self.load()

    def load(self):
        self.datasets = {}
        self.data_dictionaries = {}
        self.prompt_library = {}
        self.definitions = {}

        # Loading datasets and data dicts
        f_names = [i.split('.')[0] for i in os.listdir(f'{self.root}/data dictionaries')]
        for f_name in f_names:
            with open(f'{self.root}/queries/{f_name}.sql') as f:
                self.datasets[f_name] = pd.read_sql(f.read(), engine)
            with open(f'{self.root}/data dictionaries/{f_name}.json') as f:
                self.data_dictionaries[f_name] = json.load(f)    

        # Loading definitions
        definition_groups = os.listdir(f'{self.root}/definitions')
        for file_name in definition_groups:
            group = file_name.split('.')[0]
            with open(f'{self.root}/definitions/{file_name}', 'r') as f:
                group_definitions = json.load(f)
                self.definitions[group] = group_definitions

        # Loading prompts
        prompt_libraries = os.listdir(f'{self.root}/prompt library')
        for library in prompt_libraries:
            with open(f'{self.root}/prompt library/{library}', 'r') as f:
                prompts = json.load(f)
            library_name = library.split('.')[0]
            self.prompt_library[library_name] = prompts
        
        return
    
    def embed(self):
        pass

if __name__ == '__main__':
    context = contextManager('contexts')

    print(context.datasets)
    print(context.data_dictionaries)
    print(context.prompt_library)
    print(context.definitions)