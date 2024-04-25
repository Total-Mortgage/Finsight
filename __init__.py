import json
import os
import sys
import requests
import datetime
import uuid
import pandas as pd
from flask_socketio import SocketIO
from flask_session import Session
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from typing import Union
from llm_functions import dataToolsManager
from flask import Flask, render_template, request, session
from flask_restful import Api, Resource
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv


# loading env variables
#load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ['session_secret_key']
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
api = Api(app)
socketio = SocketIO(app)

conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};' \
               'SERVER=tmssql03.tmsdomain.com\\tmssql1;' \
               'DATABASE=Salesforce_DW;' \
               'UID=' + os.environ.get('db_user') + ';' \
               'PWD=' + os.environ.get('db_pass')
connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': conn_str})
engine = create_engine(connection_url)

datasets={}
data_dictionaries = {}
for f_name in ['loans', 'leads']:
    with open(f'queries/{f_name}.sql') as f:
        datasets[f_name] = pd.read_sql(f.read(), engine)
        print(datasets[f_name].memory_usage(deep=True).sum())
    with open(f'data dictionaries/{f_name}.json') as f:
        data_dictionaries[f_name] = json.load(f)    

def read_user_view() -> dataToolsManager:
    user_view = {i: pd.DataFrame(session['user_view'][i]) for i in session['user_view']}
    dataframe_manager = dataToolsManager(user_view)
    return dataframe_manager

def write_user_view() -> None:
    user_view = {i:datasets[i].to_dict('list') for i in datasets}
    session['user_view'] = user_view
    return


# defining llm tools
@tool
def select_base(dataset: str) -> str:
    '''
    Reset the current view and start anew by selecting a dataset. This will clear out all existing transformations to the current view.
    Parameters:
    - dataset (str): The name of the dataset to set as the new base for the current view. Valid options are: "loans", "leads".
    Returns:
    - str: A message indicating the result of the operation, either success or an error message.
    Example usage:
    select_base('loans')
    select_base('leads')
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.select_base(dataset)
    return resp
@tool
def get_current_date() -> str:
    '''Returns the current date as a string in the following format: %Y-%m-%d'''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.get_current_date()
    return resp
@tool
def get_current_view(n) -> list[dict]:
    '''
    Returns first n records in the current view. If n == 0 this returns the entire current view instead. 
    Only call this function as needed, often times the responses from other tools will give you enough information about the current view without you needing to explicitly see it.
    Parameters:
    - n (int): The number of rows to retrieve from the current view. If n==0 then all rows are retrieved
    Example usage:
    get_current_view(5) //get first 5 rows in the current view
    get_current_view(0) //get all rows in the current view
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.get_current_view(n)
    return resp
@tool
def filter_dataframe(column: str, condition: str, value: str) -> str:
    '''
    Filter the current view DataFrame based on a specified condition for a column.
    Parameters:
    - column (str): The name of the column to apply the filtering condition.
    - condition (str): The filtering condition. Valid options are: '==', '!=', '>', '<', '>=', '<=', 'is null', 'not null', 'contains'.
    - value (str): The value to compare against in the filtering condition.
    Example usage:
    getting rows where column is greater than 10:
    filter_dataframe('column_name', '>', 10)
    getting rows where column is after 2023-01-01:
    filter_dataframe('column_name', '>' '2023-01-01')
    getting rows where column is null:
    filter_dataframe('column_name', 'is null', '')
    getting rows where column is not null:
    filter_dataframe('column_name', 'not null', '')
    '''    
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.filter_dataframe(column, condition, value)
    return resp
@tool
def select_columns(columns: list[str]) -> str:
    '''
    Select specified columns in the current view. All other columns in the current view will be dropped.
    Parameters:
    - columns (list[str]): A list of column names to be selected in the current view.
    Example usage:
    select_columns(['column_1', 'column_2'])
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.select_columns(columns)
    return resp
@tool 
def group_dataframe(groupby: list[str], agg_columns: list[str], aggregation: str) -> str:
    '''
    Group the current view by the columns in the group by list and apply the aggregation function to all other columns
    Parameters:
    - groupby (list[str]): A list of column names by which to group the DataFrame.
    - agg_columns (list[str]): A list of column names by which to aggregate over the specified groups
    - aggregation (str): The aggregation function to apply to each group. Valid options are: 'min', 'max', 'sum', 'count', 'mean', 'median', 'std', 'nunique'.
    Example usage:
    group_dataframe(groupby=['Branch Code', 'Loan Officer'], agg_columns=['Loan Amount'], 'sum')
    group_dataframe(groupby=['Subject Property State'], agg_columns=['Credit Score'], 'mean')
    group_dataframe(groupby=['Branch Code'], agg_columns=['Subject Property State'], 'nunique')
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.group_dataframe(groupby, agg_columns, aggregation)
    return resp
@tool
def join_dataframe_to_current_view(dataset: str, col_1: str, col_2: str, how: str) -> str:
    '''
    Join the current view DataFrame with another dataset based on specified columns and join type using the relationship current_view.col_1 = dataset.col_2
    Parameters:
    - dataset (str): The name of the dataset to join with the current view.
    - col_1 (str): The column in the current view DataFrame for the join operation.
    - col_2 (str): The column in the specified dataset for the join operation.
    - how (str): The type of join to be performed. Valid options are: 'left', 'right', 'inner', 'outer', 'cross'.
    Example usage:
    join_dataframe_to_current_view('other_dataset', 'current_view_column', 'other_dataset_column', 'inner')
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.join_dataframe_to_current_view(dataset, col_1, col_2, how)
    return resp
@tool
def extract_datepart(date_column: str, date_part: str) -> str:
    '''
    Extract a specific date part (year, month, or day) from a date column.
    Parameters:
    - date_column (str): The name of the date column from which to extract the date part.
    - date_part (str): The desired date part to extract. Valid options are: 'year', 'month', 'day'.
    Example usage:
    extract_datepart('date_column_name', 'year')
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.extract_datepart(date_column, date_part)
    return resp
@tool
def create_custom_column_from_columns(col_1: str, col_2: str, operator: str, new_col_name: str) -> str:
    '''
    Create a new custom column by applying the specified operator between two existing columns.
    Parameters:
    - col_1 (str): Name of the first existing column.
    - col_2 (str): Name of the second existing column.
    - operator (str): Arithmetic or numeric comparison operator.
    - new_col_name (str): Name for the new custom column.
    Example usages:
    create_custom_column_from_columns('column_1', 'column_2', '+', 'sum_column')
    create_custom_column_from_columns('column_1', 'column_2', '<', 'bool_column')
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.create_custom_column_from_columns(col_1, col_2, operator, new_col_name)
    return resp

@tool
def create_custom_column_from_value(col_1: str, value: Union[int,float,str], operator, new_col_name) -> str:
    '''
    Create a new custom column by applying the specified operator between an existing column and a value.
    Parameters:
    - col_1 (str): Name of the existing column.
    - value (numeric): Numeric value for the operation.
    - operator (str): Arithmetic or numeric comparison operator.
    - new_col_name (str): Name for the new custom column.
    Example usages:
    create_custom_column_from_value('column_1', 10, '*', 'product_column')
    create_custom_column_from_value('column_1', 10, '==', 'bool_column')
    create_custom_column_from_value('column_1', '2023-03-30', '==', 'date_bool_column')
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.create_custom_column_from_columns(col_1, value, operator, new_col_name)
    return resp
@tool
def aggregate(aggregation) -> list[dict]:
    '''
    Aggregate the entire DataFrame using a specified aggregation function and update the current_view.
    Parameters:
    - aggregation (str): A string specifying the aggregation function (e.g. 'min', 'max', 'sum', 'count', 'mean', 'median', 'std, 'nunique')
    Example usage:
    aggregate('mean')
    Return Value:
    this function returns the current view after applying the aggregation function to all columns
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.aggregate(aggregation)
    return resp
@tool
def aggregate_column(column: str, aggregation: str) -> Union[float, str, int]:
    '''
    Aggregate a specific column in the DataFrame using a specified aggregation function. This does not affect the current view, instead it returns the aggregated value while leaving the current view unchanged
    Parameters:
    - column (str): The name of the column to be aggregated.
    - aggregation (str): A string specifying the aggregation function (e.g. 'min', 'max', 'sum', 'count', 'mean', 'median', 'std', 'nunique').
    Returns:
    - value: The aggregated value for the specified column.
    Example usage:
    aggregate_column('column_name', 'sum')
    Return Value:
    this function returns the value obtained from applying the aggregation function to the target column and does not affect the current view
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.aggregate_column(column, aggregation)
    return resp
@tool
def get_top_n(sortby: str, ascending: bool, n: int) -> list[dict]:
    '''
    Sort the current view by a specific column in ascending or descending order, then get the top n records from the current view
    Parameters:
    - sortby (str): The name of the column to sort the current view by
    - ascending (bool): A boolean indicating whether the current view will be sorted in ascending order (if true), or descending order (if false)
    - n (int): The number of records to extract from the current view after sorting. The first n will be selected
    Example usage:
    get_top_n('Loan Amount', 'False', 10)
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.get_top_n(sortby, ascending, n)
    return resp
    
# defining system prompt
system_prompt = f'''You are a data analyst for the company Total Mortgage and an expert in python pandas. Your task is to help users answer questions about their data.
It is important that your responses give a detailed outline of the steps you followed to arrive at your result. If you can not find an answer to the question, explain the steps that you took and what went wrong. 

The tools you have access to allow you to create and manipulate a view of the datasets. It is imperative that at the beginning of each request you use the select_base tool in order to reset the view and start from a clean dataset.

Again, select_base should always be the first tool that you call.

You have access to 2 different tables: loans and leads

The datasets contain information ranging from 2020-01-01 to {datetime.datetime.now().strftime('%Y-%m-%d')}

Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')}

Each record in the loans table represents a single loan and is uniquely identified by the 'Loan Number' column
Each record in the leads table represents a single lead and is uniquely identified by the 'Lead Id' column.

Please Note: that in our sales funnel a lead and an application are distinct objects. When a user asks for applications they are referring to loan records with a non null Application Date, when a user asks for leads they are referring to lead records with a non null Created Date

Here are data dictionaries for each of the tables:

loans = {json.dumps(data_dictionaries['loans'])}

leads = {json.dumps(data_dictionaries['leads'])}
'''

# collecting tools
tools = [select_base, get_current_view, select_columns, get_current_date, join_dataframe_to_current_view, filter_dataframe, group_dataframe, extract_datepart, create_custom_column_from_columns, create_custom_column_from_value, aggregate_column, get_top_n]


prompt = hub.pull('hwchase17/openai-tools-agent')

# loading chat agent
llm = ChatOpenAI(model='gpt-3.5-turbo-0125', organization='org-3xZIlg8TIVWf5J0rRTNGlIvt', temperature=.3)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/chat')
def chatwindow():
    return render_template('chat.html')

@socketio.on('disconnect')
def delete_session():
    del dataToolsManager.instances[session['session_id']]
    return

class chat(Resource):
    def post(self):
        data = request.get_json()

        try:
            messages = data['messages']
        except KeyError:
            return app.response_class(response=json.dumps({'error': 'messages not found'}), status=400, mimetype='application/json')
        
        input_message = messages.pop()

        input_data = {
            'input': input_message,
            'chat_history': [SystemMessage(system_prompt)]
        }

        for message in messages:
            if message['source'] == 'You':
                input_data['chat_history'].append(HumanMessage(message['message']))
            else:
                input_data['chat_history'].append(AIMessage(message['message']))
        
        output = agent_executor.invoke(input_data)
        intermediate_steps = output['intermediate_steps']
        final_message = output['output']
        
        if intermediate_steps:
            steps_taken_message = '''   Steps Taken: \n'''
            for step in intermediate_steps:
                steps_taken_message += '\t' + step[0].log.replace('\n', '') + '\n\t' + step[1] + '\n\n'
            return app.response_class(response=json.dumps({'source': 'AI', 'message': final_message, 'steps_taken': steps_taken_message}), status=200, mimetype='application/json')
        
        else:
            return app.response_class(response=json.dumps({'source': 'AI', 'message': final_message}), status=200, mimetype='application/json')

class auth(Resource):
    def post(self):
        data = request.get_json()

        try:
            code = data['code']
        except KeyError:
            return app.response_class(response=json.dumps({'error': 'Unauthorized'}), status=401, mimetype='application/json')
        
        auth_resp = requests.post('https://totalmortgage.my.salesforce.com/services/oauth2/token', data= {'grant_type': 'authorization_code', 'code': code, 'client_id': os.environ['client_id'], 'client_secret': os.environ['client_secret'], 'redirect_uri': 'http://localhost:5000/chat'})
        print(auth_resp.json(), file=sys.stderr)
        if auth_resp.status_code != 200:
            return app.response_class(response=json.dumps({'error': 'Unauthorized'}), status=401, mimetype='application/json')
        else: 
            resp_data = auth_resp.json()
            access_token = resp_data['access_token']
            resp = requests.get(resp_data['id'], headers={'Authorization': 'Bearer ' + access_token})

            user_id = resp.json()['user_id']

            resp = requests.get('https://totalmortgage.my.salesforce.com/services/data/v60.0/sobjects/User/' + user_id, headers={'Authorization': 'Bearer ' + access_token})
            branch_code = resp.json()['Branch_Code__c']

            user_view = {i:datasets[i][datasets[i]['Branch Code'] == branch_code] for i in datasets}
            session_id = uuid.uuid4()
            session['session_id'] = session_id
            dataToolsManager(user_view, session_id)
            return app.response_class(status=200, mimetype='application/json')

# adding api resources
api.add_resource(chat, '/api/chat')
api.add_resource(auth, '/api/auth')

if __name__ == '__main__':
    socketio.run(app)

