import pandas as pd
import datetime
from datetime import date

class dataToolsManager:
    instances = {}
    def __init__(self, data_dictionaries,  datasets, sid, user) -> None:
        self.datasets = datasets
        self.data_dictionaries = data_dictionaries
        self.user = user

        self.convert_dates()
        self.current_view = None
        self.base_name = None
        self.session_id = sid
        dataToolsManager.instances[self.session_id] = self
    
    def convert_dates(self):
        for dataset_name in self.datasets:
            for column in self.datasets[dataset_name].columns:
                if 'Date' in column:
                    print(f'Converting {column} to date')
                    self.datasets[dataset_name][column] = pd.to_datetime(self.datasets[dataset_name][column])

    def convert_input_type(self, col, value):
        col_dtype = self.current_view[col].dtype

        converted_value = None

        if pd.api.types.is_integer_dtype(col_dtype):
            converted_value = int(value)
        elif pd.api.types.is_float_dtype(col_dtype):
            converted_value = float(value)
        elif pd.api.types.is_bool_dtype(col_dtype):
            converted_value = bool(value)
        elif pd.api.types.is_string_dtype(col_dtype):
            converted_value = str(value)
        elif pd.api.types.is_datetime64_any_dtype(col_dtype) or 'Date' in col:
            converted_value = pd.to_datetime(value)
        else:
            print('Here')
            converted_value = value

        return converted_value

    def select_base(self, dataset):

        if dataset not in self.datasets:
            return 'There was an error, Message: {dataset} is not a valid dataset name it must be in ("loans", "leads")' 
        
        self.current_view = self.datasets[dataset]
        self.base_name = dataset

        return f'view reset to {dataset}. The dataset contains {dataset} information ranging from 2020-01-01 to {datetime.datetime.now().strftime('%Y-%m-%d')}'
    
    def get_current_date(self):
        return date.today().strftime('%Y-%m-%d')
    
    def get_current_view(self, start,  n) -> pd.DataFrame:
        return 'Current View: \n' + str(self.current_view.iloc[start: start + n, :].to_dict(orient='records')) + f'\n\n There Are {len(self.current_view)} total rows in the current view'
        
    def get_current_schema(self):
        data_dictionary = self.data_dictionaries[self.base_name]
        cols = self.current_view.columns
        schema = {}
        for item in cols:
            schema[item] = data_dictionary[item]
        return f'Here is the schema of the current view: \n\n{schema}'

    def filter_dataframe(self, column: str, condition: str, value: str) -> None:
        if column not in self.current_view.columns:
            return 'There was an error, Message: ' + column + ' is not a valid column in the current view'
        if value != '' and pd.notna(value):
            converted_value = self.convert_input_type(column, value)
        else:
            converted_value = value
        operators = {
            '==': lambda x: x == converted_value,
            '!=': lambda x: x != converted_value,
            '>': lambda x: x != None and x > converted_value,
            '<': lambda x: x != None and x < converted_value,
            '>=': lambda x: x != None and x >= converted_value,
            '<=': lambda x: x != None and x <= converted_value,
            'contains': lambda x: str(x).lower().find(converted_value.lower()) != -1,
            'is_null': lambda x: pd.isna(x),
            'not_null': lambda x: not pd.isna(x)
        }
        
        if condition not in operators:
            return 'There was an error, Message: "' + condition + '" is not a valid condition. it must be one of: "==", "!=", ">", "<", ">=", "<=", "contains", "is_null", "not_null" '

        condition_operator = operators[condition]

        self.current_view = self.current_view[self.current_view[column].apply(condition_operator)]
        return f'The current view was filtered successfully. There are {len(self.current_view)} {self.base_name} remaining in the current view'

    def select_columns(self, columns: list[str]):
        for col in columns:
            if col not in self.current_view.columns:
                return 'There was an error, Message: ' + col + ' is not in the current view'
        self.current_view = self.current_view[columns]
        return f'Success, the remaining columns in the current view are: {self.current_view.columns}. Continue transforming the current view, or use the get_current_view tool to view the selected data'

    def aggregate_group(self, groupby: list[str], agg_columns: list[str], aggregation: str):
        for col in groupby:
            if col not in self.current_view.columns:
                return 'There was an error, Message: ' + col + ' in groupby is not in the current view'
        for col in agg_columns:
            if col not in self.current_view.columns:
                return 'There was an error, Message: ' + col + ' in agg_columns is not in the current view'
        if aggregation not in ('min', 'max', 'sum', 'count', 'mean', 'median', 'std', 'nunique'):
            return 'There was an error, Message: ' + aggregation + ' is an invalid aggregation. it must be one of "min", "max", "sum", "count", "mean", "median", "std", "nunique"'

        self.current_view = self.current_view[[*groupby, *agg_columns]].groupby(by=groupby).agg(aggregation).reset_index()
        
        
        return f'Success, the current view has been grouped by: {groupby}, and the columns: {agg_columns} have been aggregated using the {aggregation} function. The remaining columns in the current view are: {self.current_view.columns}. Continue transforming the current view, or use the get_top_n function to retrieve the desired results.'

    def join_loans_to_leads(self):
        if 'Lead Id' not in self.current_view.columns:
            return 'There was an error, Message: Lead id is not in the current view, try selecting the leads dataset as the base and then call this function again'
        
        joined = self.current_view.join(self.datasets['accounts'].set_index('Attributed Lead Id'), how='inner', on='Lead Id').reset_index().join(self.datasets['loans'].drop(['Branch Code', 'Down Payment', 'DTI', 'Intended Property Use'], axis=1).set_index('Primary Account Id'), how='inner', on='Account Id').reset_index()

        for col in joined.columns:
            if joined[col].dtype == 'object':
                joined[col] = joined[col].astype(str)

        joined = joined.groupby('Lead Id').agg({
            'Lead Cost': 'max',
            'Is Blend Lead': 'max',
            'Lead Source Name': 'first',
            'Created Date': 'max',
            'Loan Officer Name': 'max',
            'Branch Code': 'max',
            'Currently Renting': 'max',
            'Desired Loan Amount': 'max',
            'Desired Rate': 'max',
            'Desired Rate Type': 'max',
            'Desired Loan Term': 'max',
            'Down Payment': 'max',
            'DTI': 'max',
            'DTI Decision': 'max',
            'Existing Mortgage Balance': 'max',
            'Existing Mortgage Monthly Payment': 'max',
            'Existing Rate Type': 'max',
            'First Time Home Buyer': 'max',
            'Has Bankruptcy': 'max',
            'Has Foreclosure': 'max',
            'Intended Property Use': 'max',
            'Lead Type': 'max',
            'Desired Loan Purpose': 'max',
            'Is VA Eligible': 'max',
            'Needs Realtor': 'max',
            'New Home Value': 'max',
            'Property State': 'max',
            'Property City': 'max',
            'Preferred Language': 'max',
            'Quoted Rate': 'max',
            'Referral Source Details': 'max',
            'Refinance Type': 'max',
            'Is Self Employed': 'max',
            'Borrower State': 'max',
            'UTM Campaign': 'max',
            'UTM Medium': 'max',
            'UTM Source': 'max',
            'UTM Term': 'max',
            'Loan Number': 'max',
            'Loan Officer': 'max',
            'Loan Amount': 'max',
            'Started Date': 'max',
            'Pre Approval Date': 'max',
            'Has Pre Approval': 'sum',
            'Pre Approval Volume': 'sum',
            'Application Date': 'max',
            'Has Application': 'sum',
            'Application Volume': 'sum',
            'Lock Date': 'max',
            'Has Rate Lock': 'sum',
            'Lock Volume': 'sum',
            'Closing Date': 'max',
            'Is Closed': 'sum',
            'Closed Volume': 'sum',
            'Credit Score': 'max',
            'LTV': 'max',
            'Last Completed Milestone': 'max',
            'Underwriter Name': 'max',
            'Loan Processor': 'max',
            'Buyers Agent Name': 'max',
            'Sellers Agent Name': 'max',
            'Subject Property State': 'max',
            'Subject Property City': 'max',
            'Subject Property Type': 'max',
            'Subject Property Purchase Price': 'max',
            'Subject Property Appraised Value': 'max',
            'Lien Position': 'max',
            'Loan Program': 'max',
            'Loan Program Type': 'max',
            'Loan Term': 'max',
            'Loan Type': 'max',
            'Rate Type': 'max',
            'Loan Purpose': 'max',
            'Note Rate': 'max',
            'Denial Date': 'max',
            'Lead Detail': 'max',
            'Outside Lead Source': 'max',
            'Self Employed': 'max',
            'Rate Lock Expiration Date': 'max',
            'Annual Income': 'max',
            'Monthly Debts': 'max',
            'Greenlight Approval Date': 'max',
            'Referral Name': 'max',
            'Account Id': 'max',
            'Front End Net': 'sum',
            'Back End Net': 'sum',
            'Loan Level Expenses': 'sum',
            'Loan Level Revenue': 'sum',
            'Commission Expenses': 'sum',
            'Total Profit': 'sum'
        }).reset_index()

        not_matched = self.current_view[~self.current_view['Lead Id'].isin(joined['Lead Id'])]
        missing_cols = [col for col in joined.columns if col not in not_matched.columns]
        for col in missing_cols:
            not_matched[col] = None

        joined = pd.concat([not_matched, joined])

        self.current_view = joined
        return f'Success, the loans dataset has been successfully joined onto the current view. the current view now contains the following columns: {self.current_view.columns}'

    def get_lead_conversion_data(self, groupby: list[str]):
        for col in [*groupby, 'Lead Id', 'Lead Cost']:
            if col not in self.current_view.columns:
                return f'There was an error, Message: {col} is not in the current view, try selecting the leads dataset as the base and then call this function again'
        
        lead_agg_data = self.current_view[[*groupby, 'Lead Id', 'Lead Cost']].groupby(groupby).agg({'Lead Id': 'count', 'Lead Cost': 'sum'}).rename(columns={'Lead Id': 'Lead Units'})
        
        joined = self.current_view.join(self.datasets['accounts'].set_index('Attributed Lead Id'), how='inner', on='Lead Id').reset_index()
        joined = joined.join(self.datasets['loans'].drop(['Branch Code', 'Down Payment', 'DTI', 'Intended Property Use'], axis=1).set_index('Primary Account Id'), how='inner', on='Account Id').reset_index()
        
        
        joined['Pre Approval Volume'] = joined['Loan Amount'].where(joined['Pre Approval Date'].notnull(), other=None)
        joined['Greenlight Approval Volume'] = joined['Loan Amount'].where(joined['Greenlight Approval Date'].notnull(), other=None)
        joined['Application Volume'] = joined['Loan Amount'].where(joined['Application Date'].notnull(), other=None)
        joined['Lock Volume'] = joined['Loan Amount'].where(joined['Lock Date'].notnull(), other=None)
        joined['Closed Volume'] = joined['Loan Amount'].where(joined['Closing Date'].notnull(), other=None)
        joined = joined.rename(columns={'Loan Amount': 'Started Volume'})

        conversion_agg_data = joined[[*groupby, 'Started Date', 'Started Volume', 'Pre Approval Date', 'Pre Approval Volume', 'Greenlight Approval Date', 'Greenlight Approval Volume', 'Application Date', 'Application Volume', 'Lock Date', 'Lock Volume', 'Closing Date', 'Closed Volume', 'Front End Net', 'Back End Net', 'Loan Level Expenses', 'Loan Level Revenue', 'Commission Expenses', 'Total Profit']].groupby(groupby).agg({'Started Date': 'count', 'Started Volume': 'sum', 'Pre Approval Date': 'count', 'Pre Approval Volume': 'sum', 'Greenlight Approval Date': 'count', 'Greenlight Approval Volume': 'sum', 'Application Date': 'count', 'Application Volume': 'sum', 'Lock Date': 'count', 'Lock Volume': 'sum', 'Closing Date': 'count', 'Closed Volume': 'sum', 'Front End Net': 'sum', 'Back End Net': 'sum', 'Loan Level Expenses': 'sum', 'Loan Level Revenue': 'sum', 'Commission Expenses': 'sum', 'Total Profit': 'sum'})
        conversion_agg_data = conversion_agg_data.rename(columns={'Started Date': 'Started Units', 'Pre Approval Date': 'Pre Approval Units', 'Greenlight Approval Date': 'Greenlight Approval Units', 'Application Date': 'Application Units', 'Lock Date': 'Lock Units', 'Closing Date': 'Closing Units'})

        self.current_view = lead_agg_data.join(conversion_agg_data, how='left').reset_index()
        self.current_view['Total Profit'] = self.current_view['Total Profit'] + self.current_view['Lead Cost']

        return f'Success the loans dataset has been successfully joined to the current leads view and conversion measures have been calculated over the requested groupings. The current view now has the following columns: {self.current_view.columns}'

    def join_dataframe_to_current_view(self, dataset: str, col_1: str, col_2: str, how: str):
        if col_2 not in self.datasets[dataset].columns:
            return 'There was an error, Message: ' + col_2 + ' is not in ' + dataset
        if col_1 not in self.current_view.columns:
            return 'There was an error, Message: ' + col_1 + ' is not in the current view'
        if how not in ('left', 'right', 'inner', 'outer', 'cross'):
            return 'There was an error, Message: the "how" parameter is invalid. it must be one of: "left", "right", "inner", "outer", "cross"'

        self.current_view = self.current_view.join(self.datasets[dataset].set_index(col_2), how=how, on=col_1).reset_index()
        return f'{dataset} was successfully joined onto the current view'

    def extract_datepart(self, date_column: str, date_part: str):
        if date_column not in self.current_view.columns:
            return f'There was an error, Message: {date_column} is not in the current view'
        
        try:
            self.current_view[date_column] = pd.to_datetime(self.current_view[date_column])
        except ValueError:
            return 'There was an error, Message: ' + date_column + ' is not a valid datetime. this function only works on columns of type date or datetime'
        if date_part == 'year':
            self.current_view[date_column + '_' + date_part] = self.current_view[date_column].dt.year
        elif date_part == 'month':
            self.current_view[date_column + '_' + date_part] = self.current_view[date_column].dt.month
        elif date_part == 'day':
            self.current_view[date_column + '_' + date_part] = self.current_view[date_column].dt.day
        else:
            return 'There was an error, Message: ' + date_part + ' is not a valid date_part. it must be one of: "year", "month", "day"'
        return 'Date part extracted successfully into column: ' + date_column + '_' + date_part

    def create_custom_column_from_columns(self, col_1, col_2, operator, new_col_name):
        if col_1 not in self.current_view.columns:
            return col_1 + ' is not in the current view'
        if col_2 not in self.current_view.columns:
            return col_2 + ' is not in the current view'
        
        operators = {
            '+': lambda x: x[0] + x[1],
            '-': lambda x: x[0] - x[1],
            '*': lambda x: x[0] * x[1],
            '/': lambda x: x[0] / x[1],
            '==': lambda x: x[0] == x[1],
            '!=': lambda x: x[0] != x[1],
            '>': lambda x: x[0] > x[1],
            '<' : lambda x: x[0] < x[1],
            '>=': lambda x: x[0] >= x[1],
            '<=': lambda x: x[0] <= x[1]
        }
        if operator not in operators:
            return f'There was an error, Message: Invalid operator {operator}. Supported operators are: +, -, *, /, ==, !=, >, <, >=, <='

        self.current_view[new_col_name] = operators[operator]((self.current_view[col_1], self.current_view[col_2]))

        return 'custom column ' + new_col_name + ' was added to the current view successfully'

    def create_custom_column_from_value(self, col_1, value, operator, new_col_name):
        if col_1 not in self.current_view.columns:
            return col_1 + ' is not in the current view'

        converted_value = self.convert_input_type(col_1, value)

        operators = {
            '+': lambda x: x + converted_value,
            '-': lambda x: x - converted_value,
            '*': lambda x: x * converted_value,
            '/': lambda x: x / converted_value,
            '==': lambda x: x == converted_value,
            '!=': lambda x: x != converted_value,
            '>': lambda x: x > converted_value,
            '<' : lambda x: x < converted_value,
            '>=': lambda x: x >= converted_value,
            '<=': lambda x: x <= converted_value
        }

        if operator not in operators:
            return f'There was an error, Message: Invalid operator {operator}. Supported operators are: +, -, *, /, ==, !=, >, <, >=, <='

        self.current_view[new_col_name] = operators[operator](self.current_view[col_1])
        return 'custom column ' + new_col_name + ' was added to the current view successfully'

    def aggregate(self, aggregation: str):
        if aggregation not in ('min', 'max', 'sum', 'count', 'mean', 'median', 'std', 'nunique'):
            return 'There was an error, Message: ' + aggregation + ' is an invalid aggregation. it must be one of "min", "max", "sum", "count", "mean", "median", "std", "nunique"'
        self.current_view = self.current_view.agg(aggregation)
        return f'Success, here ate the top 5 rows of the current view after applying the {aggregation} function:\n {self.get_current_view(0, 5)} \n\n If you need more records use the get_current_view function'
    
    def aggregate_column(self, column: str, aggregation: str):
        if aggregation not in ('min', 'max', 'sum', 'count', 'mean', 'median', 'std', 'nunique'):
            return 'There was an error, Message: ' + aggregation + ' is an invalid aggregation. it must be one of "min", "max", "sum", "count", "mean", "median", "std", "nunique"'
        
        if column not in self.current_view.columns:
            return 'There was an error, Message: ' + column + ' is not in the current view'
        
        value = self.current_view[column].agg(aggregation)
        return f'The value obtained from applying {aggregation} to {column} is: {value}'
    
    def get_top_n(self, sortby: str, ascending: bool, n: int):
        self.current_view = self.current_view.sort_values(by=sortby, ascending=bool(ascending)).head(n)
        return f'The top {n} values ranked by {sortby} with ascending={ascending} are: \n{self.get_current_view(0, n)}'
    
    