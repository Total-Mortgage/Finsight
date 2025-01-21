import json
import time
import os
import sys
import requests
import datetime
import uuid
import boto3
from flask_socketio import SocketIO
from flask_session import Session
from typing import Union
from llm_functions import dataToolsManager
from flask import Flask, render_template, request, session
from flask_restful import Api, Resource
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv
from load_context import contextManager

# loading env variables
load_dotenv()

app = Flask(__name__)

app.secret_key = os.environ['session_secret_key']
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
api = Api(app)
socketio = SocketIO(app)

context_manager = contextManager('contexts')

# def read_user_view() -> dataToolsManager:
#     user_view = {i: pd.DataFrame(session['user_view'][i]) for i in session['user_view']}
#     dataframe_manager = dataToolsManager(user_view)
#     return dataframe_manager

# def write_user_view() -> None:
#     user_view = {i:datasets[i].to_dict('list') for i in datasets}
#     session['user_view'] = user_view
#     return


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

def reset_view() -> str:
    '''
    Reset the current view and start from scratch. After calling this function you must select a new base dataset using the select_base function
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    dataframe_manager.select_base('loans')
    return 'Dataset reset successfully'

@tool
def get_current_schema() -> dict:
    '''
    Retrieve the data dictionary for this dataset, and the list of available columns in the current view 
    Parameters:
    - dataset (str): The name of the dataset to set as the new base for the current view. Valid options are: "loans", "leads".
    Returns:
    - dict: A dictionary of the form {data_dictionary, current_columns}. where data_dictionary is of the form: {column_name: {type, description, aliases}}, and current_columns is a list of string column names
    Example usage:
    get_current_schema()
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.get_current_schema()
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
    - condition (str): The filtering condition. Valid options are: '==', '!=', '>', '<', '>=', '<=', 'contains', 'is_null', 'not_null'.
    - value (str): The value to compare against in the filtering condition.
    Example usage:
    getting rows where column is greater than 10:
    filter_dataframe('column_name', '>', 10)
    
    getting rows where column is after 2023-01-01:
    filter_dataframe('column_name', '>' '2023-01-01')
    
    getting non null records
    filter_dataframe('column_name', 'not_null', '')
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
def aggregate_group(groupby: list[str], agg_columns: list[str], aggregation: str) -> str:
    '''
    Group the current view by the columns in the group by list and apply the aggregation function to all other columns. 
    WARNING: This operation edits the current_view in place and will drop columns
    - groupby (list[str]): A list of column names by which to group the DataFrame.
    - agg_columns (list[str]): A list of column names by which to aggregate over the specified groups
    - aggregation (str): The aggregation function to apply to each group. Valid options are: 'min', 'max', 'sum', 'count', 'mean', 'median', 'std', 'nunique'.
    Example usage:
    group_dataframe(groupby=['Branch Code', 'Loan Officer'], agg_columns=['Loan Amount'], 'sum')
    group_dataframe(groupby=['Subject Property State'], agg_columns=['Credit Score'], 'mean')
    group_dataframe(groupby=['Branch Code'], agg_columns=['Subject Property State'], 'nunique')
    '''
    dataframe_manager = dataToolsManager.instances[session['session_id']]
    resp = dataframe_manager.aggregate_group(groupby, agg_columns, aggregation)
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

here are data dictionaries for each of the respective datasets

loans: {context_manager.data_dictionaries['loans']}

leads: {context_manager.data_dictionaries['leads']}
'''

# collecting tools
tools = [select_base, get_current_view, get_current_schema, select_columns, get_current_date, filter_dataframe, aggregate_group, extract_datepart, create_custom_column_from_columns, create_custom_column_from_value, aggregate_column, get_top_n]


prompt = hub.pull('hwchase17/openai-tools-agent')

# loading chat agent
llm = ChatOpenAI(model='gpt-3.5-turbo-0125', organization='org-3xZIlg8TIVWf5J0rRTNGlIvt', temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, )

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/chat')
def chatwindow():
    return render_template('chat.html')

@socketio.on('disconnect')
def delete_session():
    try:
        del dataToolsManager.instances[session['session_id']]
    except KeyError:
        print(f'{session['session_id']} not found in data manager')
    return

# Initialize the CloudWatch Logs client
cloudwatch_client = boto3.client('logs')

# Define your log group and log stream names
LOG_GROUP_NAME = "finsight-messages"

def log_to_cloudwatch(message, user_id, session_id):
    """Log a message to CloudWatch in the format 'user_id/session_id/timestamp'."""
    try:
        # Construct log stream name using user_id, session_id, and a timestamp
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        log_stream_name = f"{user_id}/{session_id}"

        # Ensure the log stream exists
        response = cloudwatch_client.describe_log_streams(
            logGroupName=LOG_GROUP_NAME,
            logStreamNamePrefix=log_stream_name
        )
        log_streams = response.get('logStreams', [])
        if not log_streams:
            # Create the log stream if it doesn't exist
            cloudwatch_client.create_log_stream(
                logGroupName=LOG_GROUP_NAME,
                logStreamName=log_stream_name
            )
            sequence_token = None
        else:
            sequence_token = log_streams[0].get('uploadSequenceToken', None)

        # Log the message
        log_event = {
            'logGroupName': LOG_GROUP_NAME,
            'logStreamName': log_stream_name,
            'logEvents': [{
                'timestamp': timestamp,  # Current time in milliseconds
                'message': message
            }]
        }
        if sequence_token:
            log_event['sequenceToken'] = sequence_token

        cloudwatch_client.put_log_events(**log_event)
    except Exception as e:
        print(f"Error logging to CloudWatch: {e}")


class chat(Resource):
    def post(self):
        data = request.get_json()

        try:
            messages = data['messages']
        except KeyError:
            return app.response_class(response=json.dumps({'error': 'messages not found'}), status=400, mimetype='application/json')

        input_message = messages.pop()
        user = dataToolsManager.instances[session['session_id']].user

        input_data = {
            'input': input_message,
            'chat_history': [SystemMessage(system_prompt + f'\n\n Here is the current user\'s information: {user}')]
        }

        for message in messages:
            if 'source' in message and message['source'] == 'You':
                input_data['chat_history'].append(HumanMessage(message['message']))
            else:
                if 'tool_calls' in message:
                    input_data['chat_history'].append(AIMessage(message['message'], tool_calls=message['tool_calls'], tool_messages=message['tool_messages']))
                    for call in message['tool_calls']:
                        input_data['chat_history'].append(ToolMessage(message['tool_messages'][call['id']], tool_call_id=call['id']))
                else:
                    input_data['chat_history'].append(AIMessage(message['message']))

        print(input_data, file=sys.stderr)
        output = agent_executor.invoke(input_data)
        intermediate_steps = output['intermediate_steps']
        final_message = output['output']
        reset_view()
        # Generate steps_taken_message if intermediate_steps exist
        steps_taken_message = ''
        tool_calls = []
        tool_messages = {}
        if intermediate_steps:
            steps_taken_message = '''Steps Taken:\n'''
            tool_calls = intermediate_steps[0][0].message_log[0].tool_calls
            for step in intermediate_steps:
                steps_taken_message += '\t' + step[0].log.replace('\n', '') + '\n\t' + step[1] + '\n\n'
                tool_messages[step[0].tool_call_id] = step[0].log.replace('\n', '') + '\n\t' + step[1] + '\n\n'
        # Log messages to CloudWatch
        user_id = session['user_id']
        session_id = session['session_id']
        log_to_cloudwatch(f"Input Message: {input_message['message']}", user_id, session_id)
        if steps_taken_message:
            log_to_cloudwatch(f"Steps Taken Message: {steps_taken_message}", user_id, session_id)
        log_to_cloudwatch(f"Final Message: {final_message}", user_id, session_id)

        # Respond to the client
        if steps_taken_message:
            return app.response_class(response=json.dumps({'source': 'AI', 'message': final_message, 'steps_taken': steps_taken_message, 'tool_calls': tool_calls, 'tool_messages': tool_messages}), status=200, mimetype='application/json')
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
            resp_data = resp.json()
            print(resp_data, file=sys.stderr)
            branch_code = resp_data['Branch_Code__c']

            if branch_code == 'DEV':
                user_view = context_manager.datasets
            else:
                user_view = {i:context_manager.datasets[i][context_manager.datasets[i]['Branch Code'] == branch_code] for i in context_manager.datasets}

            session_id = uuid.uuid4()
            session['session_id'] = session_id
            session['user_id'] = user_id

            user = {
                'name': resp_data['Name'],
                'branch_code': branch_code,
                'nmls': resp_data['NMLS_Id__c']
            }
            dataToolsManager(context_manager.data_dictionaries, user_view, session_id, user)
            return app.response_class(status=200, mimetype='application/json')

# adding api resources
api.add_resource(chat, '/api/chat')
api.add_resource(auth, '/api/auth')

if __name__ == '__main__':
    socketio.run(app)

