import tkinter as tk
from tkinter import messagebox
import time
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import mysql.connector
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
stock_entry = None
second_stock_entry = None

db = mysql.connector.connect(
    host="host",
    user="username",
    password="db_passwd",
    database="stocks"
)

cursor = db.cursor()
def check_user_credentials(email, password):
    cursor.execute("SELECT * FROM test WHERE email = %s AND password = %s", (email, password))
    result = cursor.fetchone()
    return result is not None

def login():
    name = name_entry.get()
    email = email_entry.get()
    password = password_entry.get()

    if not name or not email or not password:
        messagebox.showerror("Error", "Please fill in all fields")
    else:
        # Check user credentials in the database
        if check_user_credentials(email, password):
            open_stock_window(name)
        else:
            messagebox.showerror("Error", "HAHAHA,its wrong!!!")

def open_stock_window(username):
    global stock_entry, second_stock_entry
    stock_window = tk.Toplevel(root)
    stock_window.title("Stock Selection")
    stock_window.geometry("500x500")
    stock_window.configure(bg="black")

    def print_stock_symbol():
        stock_symbol = stock_entry.get()

        if not stock_symbol:
            messagebox.showerror("Error", "Please enter a stock symbol.")
            return

        stock = yf.Ticker(stock_symbol)
        stock.info
        lst = []
        df = stock.history(period="40mo")
        df.reset_index(level=0, inplace=True)
        df = df[['Date', 'Close']]
        convert_date = df['Date']

        def convert(x):
            x = str(x).split()[0]
            return x

        converted_date = convert_date.apply(convert)
        df['Date'] = converted_date
        def convert(x):
            x = str(x).split()[0]
            return x

        converted_date = convert_date.apply(convert)
        df['Date'] = converted_date

        def str_to_datetime(s):
            split = s.split('-')
            year, month, day = int(split[0]), int(split[1]), int(split[2])
            return datetime.datetime(year=year, month=month, day=day)

        df['Date'] = df['Date'].apply(str_to_datetime)
        df

        start_date = df['Date'].iloc[4]
        latest_date = df['Date'].iloc[-1]
        day_high_5 = max(df['Close'].tail())
        day_low_5 = min(df['Close'].tail())
        print(day_high_5, day_low_5)

        df.index = df.pop('Date')
        df

        plt.rcParams["figure.figsize"] = [15, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.plot(df.index, df['Close'])
        result_window = tk.Toplevel(stock_window)
        result_window.title("Stock Analysis")
        result_window.geometry("500x150")
        result_window.configure(bg="black")

        tk.Label(result_window, text=f"Highest Value(5days): {day_high_5}", font=("Arial", 12), fg="#00D2BE", bg="black").pack(
            pady=10)
        tk.Label(result_window, text=f"Lowest Value(5days): {day_low_5}", font=("Arial", 12), fg="#00D2BE", bg="black").pack(
            pady=10)

        def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
            first_date = str_to_datetime(first_date_str)
            last_date = str_to_datetime(last_date_str)

            target_date = first_date

            dates = []
            X, Y = [], []
            last_time = False
            while True:
                df_subset = dataframe.loc[:target_date].tail(n + 1)

                if len(df_subset) != n + 1:
                    print(f'Error: Window of size {n} is too large for date {target_date}')
                    return

                values = df_subset['Close'].to_numpy()
                x, y = values[:-1], values[-1]

                dates.append(target_date)
                X.append(x)
                Y.append(y)

                next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
                next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
                next_date_str = next_datetime_str.split('T')[0]
                year_month_day = next_date_str.split('-')
                year, month, day = year_month_day
                next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

                if last_time:
                    break

                target_date = next_date

                if target_date == last_date:
                    last_time = True

            ret_df = pd.DataFrame({})
            ret_df['Target Date'] = dates
            X = np.array(X)
            for i in range(0, n):
                X[:, i]
                ret_df[f'Target-{n - i}'] = X[:, i]

            ret_df['Target'] = Y

            return ret_df

        windowed_df = df_to_windowed_df(df,
                                        str(start_date).split()[0],
                                        str(latest_date).split()[0],
                                        n=3)

        def windowed_df_to_date_X_y(windowed_dataframe):
            df_as_np = windowed_dataframe.to_numpy()

            dates = df_as_np[:, 0]

            middle_matrix = df_as_np[:, 1:-1]
            X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

            Y = df_as_np[:, -1]

            return dates, X.astype(np.float32), Y.astype(np.float32)

        dates, X, y = windowed_df_to_date_X_y(windowed_df)

        dates.shape, X.shape, y.shape

        q_80 = int(len(dates) * .8)
        q_90 = int(len(dates) * .9)

        dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
        dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

        dates_val_train = np.append(dates_train, dates_val)
        y_val_train = np.append(y_train, y_val)

        model = Sequential([layers.Input((3, 1)),
                            layers.LSTM(64),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(1)])

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['mean_absolute_error'])

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

        train_predictions = model.predict(X_train).flatten()
        plt.plot(dates_train, train_predictions)
        plt.plot(dates_train, y_train)
        plt.legend(['Training Predictions', 'Training Observations'])
        val_predictions = model.predict(X_val).flatten()

        plt.rcParams["figure.figsize"] = [15, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.plot(dates_val, val_predictions)
        plt.plot(dates_val, y_val)
        plt.legend(['Validation Predictions', 'Validation Observations'])

        test_predictions = model.predict(X_test).flatten()

        plt.rcParams["figure.figsize"] = [15, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.plot(dates_test, test_predictions)
        plt.plot(dates_test, y_test)
        plt.legend(['Testing Predictions', 'Testing Observations'])

        plt.rcParams["figure.figsize"] = [15, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.plot(dates_train, train_predictions)
        plt.plot(dates_train, y_train)
        plt.plot(dates_val, val_predictions)
        plt.plot(dates_val, y_val)
        plt.plot(dates_test, test_predictions)
        plt.plot(dates_test, y_test)
        plt.legend(['Training Predictions',
                    'Training Observations',
                    'Validation Predictions',
                    'Validation Observations',
                    'Testing Predictions',
                    'Testing Observations'])
        # predicting next day
        trial_vals_lst = df['Close'].tail(3).tolist()
        trial_vals_np = np.array(trial_vals_lst)
        trial_vals = trial_vals_np.reshape((1, 3, 1))
        tensor1 = tensorflow.convert_to_tensor(trial_vals)

        predicted = model.predict(tensor1).flatten()
        print(predicted)
        print(df)
        predicted_date = latest_date + datetime.timedelta(days=1)
        print(predicted_date)
        plt.rcParams["figure.figsize"] = [15, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.plot(dates_train, train_predictions)
        plt.plot(dates_train, y_train)
        plt.plot(dates_val, val_predictions)
        plt.plot(dates_val, y_val)
        plt.plot(dates_test, test_predictions)
        plt.plot(dates_test, y_test)
        plt.plot(predicted_date, predicted, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        plt.show()  # This line will display the plot
        plt.legend(['Training Predictions',
                    'Training Observations',
                    'Validation Predictions',
                    'Validation Observations',
                    'Testing Predictions',
                    'Testing Observations',
                    'Next Day Prediction'])

        print('lol')

    tk.Label(stock_window, text=f"Welcome, {username}!", font=("Arial", 16, "bold"), fg="#00D2BE", bg="black").pack(
        pady=10)

    clock_label = tk.Label(stock_window, font=("Arial", 14), fg="white", bg="black")
    clock_label.pack()

    update_clock(clock_label)

    stock_entry = tk.Entry(stock_window, font=("Arial", 12), bd=3, relief=tk.GROOVE)
    stock_entry.pack(pady=10)

    check_stock_button = tk.Button(stock_window, text="Check Stock",
                                   command=print_stock_symbol,
                                   font=("Arial", 12), bg="#252525", fg="white", padx=10, pady=5)
    check_stock_button.pack(pady=10)

    second_stock_entry = tk.Entry(stock_window, font=("Arial", 12), bd=3, relief=tk.GROOVE)
    second_stock_entry.pack(pady=5)

    compare_stocks_button = tk.Button(stock_window, text="Compare Stocks",
                                      command=compare_stocks,
                                      font=("Arial", 12), bg="#252525", fg="white", padx=10, pady=5)
    compare_stocks_button.pack(pady=10)


def compare_stocks():
    global stock_entry, second_stock_entry

    stock1 = stock_entry.get()
    stock2 = second_stock_entry.get()

    if not stock1 or not stock2:
        messagebox.showerror("Error", "Please enter both stock symbols.")
        return

    try:
        stock_info_1 = get_stock_info(stock1)
        stock_info_2 = get_stock_info(stock2)

        # Load models for each stock
        model1 = load_your_model()
        model2 = load_your_model()

        # Calculating next day predictions
        predicted_1 = get_next_day_prediction(stock_info_1, model1)
        predicted_2 = get_next_day_prediction(stock_info_2, model2)

        # Plotting stock values and predictions
        plot_stock_comparison(stock1, stock_info_1, predicted_1, stock2, stock_info_2, predicted_2)

        # Display predictions
        #messagebox.showinfo("Next Day Prediction", f"Next Day Prediction for {stock1}: {predicted_1}\n"
                                                   #f"Next Day Prediction for {stock2}: {predicted_2}")

    except Exception as e:
        print("Error:", e)
        messagebox.showerror("Error", f"Error comparing stocks: {e}")

def get_next_day_prediction(stock_info, model):
    # Assuming the last three days' closing values for prediction
    trial_vals_lst = stock_info['Close'].tail(3).tolist()
    trial_vals_np = np.array(trial_vals_lst)
    trial_vals = trial_vals_np.reshape((1, 3, 1))

    predicted = model.predict(trial_vals).flatten()

    predicted = np.clip(predicted, 0, None)

    return predicted[0]
def load_your_model():
    model = Sequential([layers.Input((3, 1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    return model

def plot_stock_comparison(stock1, stock_info_1, predicted_1, stock2, stock_info_2, predicted_2):
    plt.rcParams["figure.figsize"] = [15, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Plotting stock values
    plt.plot(stock_info_1['Date'], stock_info_1['Close'], label=f'{stock1.upper()} Closing Price')
    plt.scatter(stock_info_1['Date'].iloc[-1], stock_info_1['Close'].iloc[-1], color='red', marker='*', s=100,
                label=f'Next Day Prediction for {stock1.upper()}')

    plt.plot(stock_info_2['Date'], stock_info_2['Close'], label=f'{stock2.upper()} Closing Price')
    plt.scatter(stock_info_2['Date'].iloc[-1], stock_info_2['Close'].iloc[-1], color='blue', marker='*', s=100,
                label=f'Next Day Prediction for {stock2.upper()}')

    plt.title("Stock Price Comparison")
    plt.xlabel("Date")
    plt.ylabel("Close Price")

    # Set legend position to upper right
    plt.legend(loc='upper left')

    plt.show()

def get_stock_info(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period="40mo")
    df.reset_index(level=0, inplace=True)
    df = df[['Date', 'Close']]

    convert_date = df['Date']
    def convert(x):
        x = str(x).split()[0]
        return x

    converted_date = convert_date.apply(convert)
    df['Date'] = converted_date

    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    df['Date'] = df['Date'].apply(str_to_datetime)

    return df
def update_clock(label):
    current_time = time.strftime("%H:%M:%S")
    label.config(text=f"Current Time: {current_time}")
    label.after(1000, lambda: update_clock(label))


def check_stocks(stock):
    if not stock:
        messagebox.showerror("Error", "Please enter a stock symbol.")
        return

    try:
        # Call the function from main.py
        result = main.get_stock_info(stock)

        print("Function result:", result)

        # Display the output
        messagebox.showinfo("Stock Info", result)
    except Exception as e:
        print("Error:", e)
        messagebox.showerror("Error", f"Error checking stock: {e}")


root = tk.Tk()
root.title("Stock Market Explorer")
root.geometry("500x500")
root.configure(bg="#252525")

tk.Label(root, text="Stock Market Explorer", font=("Arial", 20, "bold"), fg="#00D2BE", bg="#252525").grid(row=0,
                                                                                                          column=0,
                                                                                                          columnspan=2,
                                                                                                          pady=10)
tk.Label(root, text="Name:", font=("Arial", 12), fg="white", bg="#252525").grid(row=1, column=0, padx=10, pady=5,
                                                                                sticky="e")
name_entry = tk.Entry(root, font=("Arial", 12), bd=3, relief=tk.GROOVE)
name_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Email:", font=("Arial", 12), fg="white", bg="#252525").grid(row=2, column=0, padx=10, pady=5,
                                                                                 sticky="e")
email_entry = tk.Entry(root, font=("Arial", 12), bd=3, relief=tk.GROOVE)
email_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Password:", font=("Arial", 12), fg="white", bg="#252525").grid(row=3, column=0, padx=10, pady=5,
                                                                                    sticky="e")
password_entry = tk.Entry(root, show="*", font=("Arial", 12), bd=3, relief=tk.GROOVE)
password_entry.grid(row=3, column=1, padx=10, pady=5)

login_button = tk.Button(root, text="Login", command=login, font=("Arial", 14), bg="#252525", fg="white", padx=10,
                         pady=5)
login_button.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()