import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from faker import Faker

def filter_dataframe(df: pd.DataFrame, column_name: str, threshold: float) -> pd.DataFrame:
    """
    Filters the DataFrame based on a threshold in a specific column.

    • Input: 
    - A DataFrame  
    - Name of a column (str)
    - Threshold (float)

    • Output: 
    - A filtered DataFrame
    """
    filtered_df = df[df[column_name] > threshold]
    return filtered_df

def generate_regression_data(n_samples: int):
    """
    Simulate a dataset for a regression problem.

    • Input: 
    - n_samples 

    • Output: 
    - X DataFrame dependent variable
    - Y Series independent varaible
    """

    if n_samples <= 0:
        return None
    
    fake = Faker()

    # Independent variables
    data = {
        'fe1': [fake.random_number(digits=3) for _ in range(n_samples)],
        'fe2': [fake.random_number(digits=3) for _ in range(n_samples)],
        'fe3': [fake.random_number(digits=3) for _ in range(n_samples)],
    }

    X = pd.DataFrame(data)

    # Dependent variable
    noise = np.random.normal(0, 10, n_samples)  
    y = 1.8 * X['fe1'] + 2.5 * X['fe2'] + 2.2 * X['fe3'] + noise 

    return X, pd.Series(y)

def train_multiple_linear_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """
    Train a multiple linear regression model.

    • Input: 
    - X DataFrame independent variables.
    - y Series dependent variable.

    • Output: 
    - Linear regression model.
    """
    model = LinearRegression()

    model.fit(X, y)

    return model

def flatten_list(list_of_list: list) -> list:
    """
    Flatten a list of lists into a list
    • Input: 
    - list of list
    • Output: 
    - Flaten list
    """
    flat_list = []
    for sublist in list_of_list:  
        for item in sublist:     
            flat_list.append(item) 
    return flat_list
def group_and_aggregate(df: pd.DataFrame, group_column: str, aggregate_column: str) -> pd.DataFrame:
    """
    Group a DataFrame by a specified column and calculate the mean of another column.

    • Input: 
    - df 
    - group_column, column to group by.
    - aggregate_column, column to aggregate.

    • Output: 
    - A DataFrame with the grouped and aggregated values.
    """

    grouped_dataframe = df.groupby(group_column)

    agregated_column = grouped_dataframe[aggregate_column].mean()
    
    agregated_dataframe = agregated_column.reset_index()

    return agregated_dataframe

def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """
    Train a logistic regression model on binary data.

    • Input: 
    - X DataFrame independent variables.
    - y Series dependent variable.

    • Output: 
    - Logistic regression model.
    
    """
    model = LogisticRegression()

    model.fit(X, y)

    return model

def apply_function_to_column(df: pd.DataFrame, column_name: str, func) -> pd.DataFrame:
    """
    Apply a custom function to each value in a specified column .

    • Input: 
    - df 
    - column_name 
    - function to apply to each value in the specified column.

    • Output: 
    - pd.DataFrame: The modified DataFrame with the updated column.
    """
    df[column_name] = df[column_name].apply(func)
    return df

def quadratic_function(x) -> float:
    """
    Calculate the value of the quadratic function: 2x^2 + 3x + 5.

    • Input: 
    - x, input value.

    • Output: 
    - calculated value.
    """
    calculated_value = 2 * (x ** 2) + 3 * x + 5
    return calculated_value

def filter_and_square(numbers) -> list:
    """
    Filter the numbers mayor to five and calculate squeare.

    • Input: 
    - numbers

    • Output: 
    - Filtered and square numbers
    """
    square_numbers = [] 
    for x in numbers:
        if x>5:
            square_numbers.append(x**2)
    return square_numbers

# EXERCISE 1
# Load the DataFrame (the first column is set as the index)
dataFrame = pd.read_csv("bostonHousing.csv", index_col=0)
df_subset = dataFrame.head(100)
result = filter_dataframe(df_subset, 'indus', 7.88)
# print(result)

# EXERCISE 2
# Generate a dataset for a regression problem with a specified number of samples
X, Y = generate_regression_data(10)
# print(X)  
# print()
# print(Y)  

# EXERCISE 3
# Train a multiple linear regression model using the generated data
m_linear_regresion = train_multiple_linear_regression(X, Y)
# print("Coefficients:", m_linear_regresion.coef_)
# print("Intercept:", m_linear_regresion.intercept_)

# EXERCISE 4
# Flatten a list of lists into a single list
list_of_list = [[2, 6], [1, 0], [9, 3]]
f_list = flatten_list(list_of_list)
# print(f_list)

# EXERCISE 5
# Group the DataFrame by the 'class' column and calculate the mean of the 'number' column
dataSet = {'class': ['A', 'B', 'A', 'B', 'A'],'number': [10, 20, 30, 40, 50]}
df = pd.DataFrame(dataSet)
result = group_and_aggregate(df, 'class', 'number')
# print(result)

# EXERCISE 6
# Load the dataset and split into training and testing sets
df_logist_regresion = pd.read_csv("petAdoption.csv")
X = df_logist_regresion.iloc[:, :-1]
Y = df_logist_regresion.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
logistic_regression = train_logistic_regression(X_train, Y_train)
predictions = logistic_regression.predict(X_test)
# print(predictions)

# EXERCISE 7
# Load the DataFrame and apply a quadratic function to a specified column
dataFrame = pd.read_csv("bostonHousing.csv", index_col=0)
# Take the first one hundred rows
df_subset = dataFrame.head(50)
modified_df = apply_function_to_column(df_subset, 'zn', quadratic_function)
# print(modified_df)

# EXERCISE 8
# Filter a list of numbers greater than a threshold and calculate their square
input_list = [1, 4, 12, 3, 9, 21, 28, 2]
result = filter_and_square(input_list)
# print(result)




