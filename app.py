from datetime import date, datetime as dt
from io import StringIO
import streamlit as st
from csv import reader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import altair as alt
from IPython.core.pylabtools import figsize
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima.arima import CHTest

#start streamlit
#cd Vania\Python\Projects\Streamlit\mauna_loa_co2
#streamlit run app.py

#functions
def line_chart(df, col, monthly, lc, pc, rc):
    #selection
    hover = alt.selection_single(
        encodings=["x"],
        nearest=True,
        on="mouseover",
        empty="none",
        clear="mouseout"
    )

    #base chart, show value on hover, & vertical line
    if monthly == True:
        base = alt.Chart(df).mark_line(color = lc).encode(
            x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
            y=alt.Y(col + ":Q", title='Average CO2 (ppm)')
        )

        show = alt.Chart(df).mark_point().encode(
            x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
            opacity=alt.value(0),
            tooltip=["Month","Year", alt.Tooltip(col + ":Q", format=",.2f")]
        ).add_selection(
            hover
        )

        rule = alt.Chart(df).mark_rule(color = rc).encode(
            x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        ).transform_filter(
            hover
        )

        chart = alt.layer(
            base, show, rule,
            base.mark_circle(color=pc).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0))
            ).transform_filter(hover)
        ).properties(
            title="Monthly Average CO2 Concentration",
            height=400
        ).interactive()
    else:
        base = alt.Chart(df).mark_line(color = lc).encode(
            x=alt.X("Year", title="Time (Year)"),
            y=alt.Y("Avg_CO2:Q", title='Average CO2 (ppm)')
        ).transform_aggregate(
            Avg_CO2="mean("+col+")",
            groupby=["Year"]
        )

        show = alt.Chart(df).mark_point().encode(
            x='Year',
            opacity=alt.value(0),
            tooltip=["Year", alt.Tooltip("Avg_CO2:Q", format=",.2f", title = "Avg. CO2")]
        ).transform_aggregate(
            Avg_CO2="mean(" + col + ")",
            groupby=["Year"]
        ).add_selection(
            hover
        )

        rule = alt.Chart(df).mark_rule(color = rc).encode(
            x="Year",
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        ).transform_filter(
            hover
        )

        chart = alt.layer(
            base, show, rule,
            base.mark_circle(color=pc).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0))
            ).transform_filter(hover)
        ).properties(
            title="Yearly Average CO2 Concentration",
            height=400
        ).interactive()
    return chart

def combine_data(actual, column, model, periods):
    fitted, confint = model.predict(n_periods=int(periods), return_conf_int=True)
    index_of_fc = pd.date_range(actual[column].index[-1], periods=int(periods), freq='MS')
    fitted_series = pd.DataFrame(fitted, index=index_of_fc)
    fitted_series.index.name = "Time"
    fitted_series.rename(columns={0: "Forecast"}, inplace=True)

    combine = pd.concat([actual, fitted_series], axis=0)
    combine_noidx = combine.reset_index()
    combine_noidx["Month"] = combine_noidx["Time"].apply(lambda x: x.strftime("%b"))
    combine_noidx["Year"] = combine_noidx["Time"].apply(lambda x: x.strftime("%Y"))
    return combine_noidx

def forecast_plot(data, column, lc_act, lc_pred, rc, pc):
    act = alt.Chart(data).mark_line(color=lc_act).encode(
        x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
        y=alt.Y(column + ":Q", title='Average CO2 (ppm)')
    )

    pred = alt.Chart(data).mark_line(color=lc_pred).encode(
        x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
        y=alt.Y("Forecast:Q", title='Average CO2 (ppm)')
    )

    hover = alt.selection_single(
        encodings=["x"],
        nearest=True,
        on="mouseover",
        empty="none",
        clear="mouseout"
    )

    show = alt.Chart(data).mark_point().encode(
        x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
        opacity=alt.value(0),
        tooltip=[alt.Tooltip(column + ":Q", format=",.2f"), alt.Tooltip("Forecast:Q", format=",.2f"), "Month", "Year"]
    ).add_selection(
        hover
    )

    rule = alt.Chart(data).mark_rule(color=rc).encode(
        x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).transform_filter(
        hover
    )

    point_act = act.mark_circle(color=pc).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).transform_filter(hover)

    point_pred = pred.mark_circle(color=pc).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).transform_filter(hover)

    chart = alt.layer(act, pred, show, rule, point_act, point_pred).properties(
        height=400
    ).interactive()
    return chart

#configs
st.set_page_config(page_title="Atmospheric CO2", layout="wide")

#sidebar
st.sidebar.image('img/scrippslogo-removebg.png')
st.sidebar.header("Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file downloaded from Scripps CO2 website", type = "csv")
if uploaded_file is not None:
    st.sidebar.success("Upload data success!")
    st.sidebar.markdown("""
            <strong>File Details</strong>
            <table>
                <tr>
                    <th>Name</th>
                    <td><i>{a}</i></td>
                </tr>
                <tr>
                    <th>Type</th>
                    <td><i>{b}</i></td>
                </tr>
                <tr>
                    <th>Size</th>
                    <td><i>{c} Bytes</i></td>
                </tr>
            </table>
            <br>
            """.format(a=uploaded_file.name, b=uploaded_file.type, c=uploaded_file.size), unsafe_allow_html=True)
else:
    st.sidebar.info("Awaiting data upload")

#main page
st.title("Atmospheric CO2 of Mauna Loa Observatory, Hawaii")
st.write("<p><strong>Created by </strong><a href='https://github.com/avania3008' target='_blank'>Aurellia Vania Yosephine Budiman</a></p>", unsafe_allow_html=True)

if uploaded_file is not None:
    with st.expander("Show Dataset", expanded = True):
        # format data table
        header = ["Yr", "Mn", "Date1", "Date2", "CO2", "seasonally_adjusted", "fit",
              "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]

        with StringIO(uploaded_file.getvalue().decode("utf-8")) as file:
            rows = list(reader(file))
            selected_rows = rows[57:]

        df = pd.DataFrame(selected_rows)
        df.columns = header

        st.write("The raw CSV file downloaded from Scripps CO2 website contains some data informations and unstructured table headers, so I decided to format the file into a structured table just for the data. Here is the formatted table from the uploaded CSV file :")
        st.dataframe(df)

        data_download = df.to_csv(index = False).encode('utf-8')
        #download formatted table
        st.download_button(label = "Download Formatted Table", data = data_download, file_name = "mauna_loa_co2_formatted.csv", mime = 'text/csv')

        st.write("""
            
        The Atmospheric CO2 dataset contains 10 columns :

        | Column | Column Name | Description |
        | ----------- | ----------- | ----------- |
        | 1-4 | Yr, Mn, Date1, Date2 | Dates in several redundant (year, month, date) |
        | 5 | CO2 | Monthly CO2 concentrations (ppm), adjusted to 24:00 hours on the 15th of each month |
        | 6 | seasonally_adjusted | Monthly CO2 concentrations (ppm) after a seasonal adjustment (subtracting from the data a 4-harmonic fit with a linear gain factor) to remove the quasi-regular seasonal cycle |
        | 7 | fit | Smoothed version of the data generated from a stiff cubic spline function plus 4-harmonic functions with linear gain |
        | 8 | seasonally_adjusted_fit | Smoothed version with the seasonal cycle removed |
        | 9 | CO2_filled | Identical to Column 5, except the missing values from the Column 5 have been filled with values from Column 7 |
        | 10 | seasonally_adjusted_filled | Identical to Column 6, except the missing values from the Column 5 have been filled with values from Column 8 |
        **Note** : *Missing values are denoted by -99.99*
        """)

    with st.expander("Data Visualization", expanded = False):
        st.write("This part shows the line chart of the CO2 concentrations from certain range of date")
        period = st.selectbox('Select period', ["Monthly", "Yearly"])
        c1, c2 = st.columns(2)
        smoothed = c1.radio("Smoothed", ["Yes", "No"])
        seasonally = c2.radio("Seasonally Adjusted", ["Yes", "No"])

        # preprocess dataframe
        # convert year & month (int) to string
        df["Yr"] = df["Yr"].apply(lambda x: str(x))
        df["Mn"] = df["Mn"].apply(lambda x: str(x).strip())
        # rename column
        df = df.rename(columns={"Yr": "Year"})
        # convert month number to month name
        month_dict = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
                      "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
                      "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
        df["Month"] = df["Mn"].map(month_dict)
        # concat month name and year
        df["Time"] = df["Mn"] + "-" + df["Year"]
        # convert time (string type) to datetime type
        df["Time"] = df["Time"].apply(lambda x: dt.strptime(x, "%m-%Y"))

        # choose column to visualize by user input (radio button)
        if smoothed == "Yes" and seasonally == "Yes":
            col = "seasonally_adjusted_fit"
            #null_row = df[df[col] == -99.99].index
            #df[col] = df[col].replace([-99.99],df["seasonally_adjusted_filled"][null_row])
        elif smoothed == "Yes" and seasonally == "No":
            col = "fit"
            #null_row = df[df[col] == -99.99].index
            #df[col] = df[col].replace([-99.99], df["CO2_filled"][null_row])
        elif smoothed == "No" and seasonally == "Yes":
            col = "seasonally_adjusted_filled"
        else:
            col = "CO2_filled"

        # show used column for the chart
        c3, c4, c5, c6 = st.columns([10, 1, 1, 1])
        c3.info("Used column : {}".format(col))
        line_color = c4.color_picker("Line color", value = "#1f77b4")
        point_color = c5.color_picker("Point color", value = "#d62728")
        rule_color = c6.color_picker("Rule color", value = "#7f7f7f")
        if period == "Monthly" :
            start_month = c1.selectbox('Start Month', ["January", "February", "March", "April", "May", "June", "July"
                                                        , "August", "September", "October", "November", "December"])
            end_month = c2.selectbox('End Month', ["January", "February", "March", "April", "May", "June", "July"
                                                    , "August", "September", "October", "November", "December"])
            start_year = c1.slider('Start Year', min_value=1958, max_value=2021, value=2000)
            end_year = c2.slider('End Year', min_value=1958, max_value=2021, value=2021)

            std = dt.strptime((str(start_month)+"-"+str(start_year)), "%B-%Y")
            end = dt.strptime((str(end_month)+"-"+str(end_year)), "%B-%Y")

            if std > end:
                st.error("The start date mustn't be after the end date")
            else:
                time_ranged = df.loc[((df['Time'] >= std) & (df['Time'] <= end))]
                data = time_ranged[time_ranged[col] != -99.99]
                monthly = line_chart(data,col,True,line_color,point_color,rule_color)
                st.altair_chart(monthly, use_container_width=True)
        else :
            start_year = c1.slider('Start Year', min_value=1958, max_value=2021, value=2000)
            end_year = c2.slider('End Year', min_value=1958, max_value=2021, value=2021)

            #convert date to datetime
            std = dt.strptime(str(start_year), "%Y")
            end = dt.strptime(str(end_year), "%Y")

            if std > end:
                st.error("The start year mustn't be after the end year")
            else:
                time_ranged = df.loc[((df['Time'] >= std) & (df['Time'] <= end))]
                data = time_ranged[time_ranged[col] != -99.99]
                yearly = line_chart(data, col, False,line_color,point_color,rule_color)
                st.altair_chart(yearly, use_container_width=True)

    with st.expander("Analysis Results", expanded = False):
        st.subheader("1. Data Preparation")
        st.write("Importing all needed libraries for the analysis :")
        st.code("""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime as dt, date
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from pmdarima.arima import CHTest
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm""")
        st.write("""
        Using the formatted data frame, the preparations we need are :
        - Drop rows that have missing values on all column 5-10, which means first 2 rows and last 3 rows
        - Create a new column named 'Time' that was converted from 'Mn' column and 'Yr' column
        - Subset the columns into only 'Time' and a column to be analyzed (one column from column 7-10)
        - Set 'Time' column as index
        """)
        st.code("""
# import the formatted data frame
co2 = pd.read_csv('mauna_loa_co2_formatted.csv')

# drop rows with missing values
co2 = co2.iloc[2:765,]

# create 'Time' column & convert to datetime
co2['Time'] = co2['Mn'].astype(str) + '-' + co2['Yr'].astype(str)
co2['Time'] = co2['Time'].apply(lambda x: dt.strptime(x, '%m-%Y'))       

# subset the columns
co2 = co2[['Time','CO2_filled']] # in this case using 9th column

# set index
co2.set_index('Time', inplace = True)      
        """)
        col1, col2 = st.columns(2)
        select_column = col1.selectbox('Select the column to be analyzed', ['CO2_filled', 'seasonally_adjusted_filled', 'fit', 'seasonally_adjusted_fit'])
        co2 = df.iloc[2:764,]
        co2 = co2[["Time", select_column]]
        co2.set_index("Time", inplace = True)
        col1.code("co2.head()")
        col2.dataframe(co2.head())

        st.subheader("2. Check for Stationarity and Seasonality")
        st.write("""
        Next before building the model, we need to check the time series stationarity and seasonality.
        One of the way to do that is by decomposing the time series into its components, such as base level, trend, seasonality, and error.
        Depending on the nature of the trend and seasonality, a time series can be modeled as an additive or multiplicative :
                 
        - **Additive Time Series** : Value = Base Level + Trend + Seasonality + Error
        - **Multiplicative Time Series** : Value = Base Level \* Trend \* Seasonality \* Error""")
        st.code("""
# decomposing time series to its components
# Additive Decomposition
result_add = seasonal_decompose(co2['CO2_filled'], model='additive', extrapolate_trend='freq')

# Multiplicative Decomposition 
result_mul = seasonal_decompose(co2['CO2_filled'], model='multiplicative', extrapolate_trend='freq')
        """)
        result_add = seasonal_decompose(co2[select_column], model='additive', extrapolate_trend='freq')
        result_mul = seasonal_decompose(co2[select_column], model='multiplicative', extrapolate_trend='freq')
        sd_add, sd_mul = st.columns(2)

        sd_add.write("<p style='text-align:center'><strong>Additive Decomposition</strong></p>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(3, 1, figsize=(10,12))
        ax1[0].title.set_text("Trend")
        ax1[0].plot(result_add.trend)
        ax1[1].title.set_text("Seasonality")
        ax1[1].plot(result_add.seasonal)
        ax1[2].title.set_text("Residual")
        ax1[2].plot(result_add.resid)
        sd_add.pyplot(fig1)

        sd_mul.write("<p style='text-align:center'><strong>Multiplicative Decomposition</strong></p>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(3, 1, figsize=(10,12))
        ax2[0].title.set_text("Trend")
        ax2[0].plot(result_mul.trend)
        ax2[1].title.set_text("Seasonality")
        ax2[1].plot(result_mul.seasonal)
        ax2[2].title.set_text("Residual")
        ax2[2].plot(result_mul.resid)
        sd_mul.pyplot(fig2)

        st.write("""
        From both additive decomposition and multiplicative decomposition, it can be seen that this time series data has both trend pattern and seasonal pattern, which means the data is non-stationary.
        To be certain, let's check the stationary test using Augmented Dickey Fuller (ADF) test or Unit Root Test.
        
        **ADF Test**\n
        *H0 : The time series is non-stationary*\n
        *H1 : The time series is stationary*\n
        *Reject null hypothesis if p-value test is less than the significance level (0.05)*
        """)
        st.code("""
adf_test = adfuller(co2["CO2_filled"].values, autolag='AIC')
print(f'ADF Statistic: {adf_test[0]}')
print(f'P-Value: {adf_test[1]}')
        """)
        adf_test = adfuller(co2[select_column].values, autolag='AIC')
        adf, pval = st.columns(2)
        adf.metric(label="ADF Statistic", value=adf_test[0])
        pval.metric(label="P-Value", value=adf_test[1])
        st.write("""
        From the above results, the p-value from ADF test is 1, which is more than the significance level (0.05), therefore the null hypothesis isn't rejected (the time series is non-stationary). 
        To make non-stationary into stationary for effective time series modeling, one of the popular methods is by **differencing**.
        
        One of most popular models for time series forecasting is ARIMA (Auto Regressive Integrated Moving Average). Differencing can be done alongside we build the ARIMA model. 
        Also, we need to notice that ARIMA model doesn't support seasonality, therefore we can use SARIMA (Seasonal ARIMA) as an alternative.
        """)
        st.subheader("3. ACF Plot and PACF Plot")
        st.write("""
        Before building the appropriate model, let's see its ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots.
        """)
        acf, pacf = st.columns(2)
        acf.pyplot(plot_acf(co2[select_column], title = "Autocorrelation of "+ str(select_column)))
        pacf.pyplot(plot_pacf(co2[select_column], title="Partial Autocorrelation of " + str(select_column)))
        st.write("""
        If the autocorrelations are positive for many number of lags (10 or more) and goes into the far negative zone slowly, then the series needs further **differencing**. 
        It indicates that the series tend to have positive trend pattern on it. On the other hand, if the autocorrelation goes into the far negative zone very quickly, then the series is probably over-differenced.
        """)

        st.subheader("4. Building ARIMA Model Manually")
        st.write("""
        ***Order Combinations of ARIMA(p,d,q):***
        - ARIMA(1,1,1)
        - ARIMA(1,1,2)
        - ARIMA(1,2,1)
        - ARIMA(2,1,1)
        - ARIMA(2,1,2)
        
        Using above combinations, the best model is the fourth model with the smallest AIC = 1656.508, also all of the variables are significant with p-value less than significance level 0.05. Here is the summary :
        """)
        st.code("""
model4 = ARIMA(co2["CO2_filled"], order=(2,1,1), freq=co2.index.inferred_freq)
model4_fit = model4.fit()
print(model4_fit.summary())
        """)
        model4 = ARIMA(co2[select_column].astype(float), order=(2, 1, 1), freq=co2.index.inferred_freq)
        model4_fit = model4.fit()
        st.write(model4_fit.summary())
        st.write("""
        With the above model, we can plot the residuals to ensure that there aren't any pattern (it means having constant mean and variance) and also predict the fit value of CO2 concentration.
        """)
        st.write("#### Residual Plot")
        residuals = pd.DataFrame(model4_fit.resid)
        g = residuals.plot(title="Residuals", ylim=[-3.5, 3.5])
        g.set_xlabel('Time')
        g.set_ylabel('Residuals')
        resid_plot = g.figure
        st.pyplot(resid_plot)
        st.code("""
# Actual vs Fitted
fitted = model4_fit.predict()
predicted = co2.copy()
predicted["Fitted"] = np.around(fitted.values, 2)
predicted.rename(columns = {"CO2_filled": "Actual"}, inplace = True)
predicted['Fitted'] = predicted['Fitted'].replace(0, np.nan)
print(predicted.head())
        """)
        fitted = model4_fit.predict()
        predicted = co2.copy()
        predicted["Fitted"] = np.around(fitted.values, 2)
        predicted.rename(columns={select_column: "Actual"}, inplace=True)
        predicted['Fitted'] = predicted['Fitted'].replace(0, np.nan)
        st.write(predicted.head())
        st.write("The residual errors seem fine with near zero mean and uniform variance, so let’s plot the actuals against the fitted values :")
        pred = predicted.reset_index().melt('Time', var_name='Type', value_name='CO2').dropna()
        hover = alt.selection_single(
            encodings=["x"],
            nearest=True,
            on="mouseover",
            empty="none",
            clear="mouseout"
        )

        # The basic line
        line = alt.Chart(pred.astype(str)).mark_line(interpolate='basis').encode(
            x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
            y=alt.Y('CO2:Q', axis=alt.Axis(title='CO2 Concentration (ppm)')),
            color='Type:N'
        )

        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(pred.astype(str)).mark_point().encode(
            x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
            opacity=alt.value(0),
            tooltip=[alt.Tooltip("Time:O", timeUnit="yearmonth", title="Time")]
        ).add_selection(
            hover
        )

        # Draw points on the line, and highlight based on selection
        points = line.mark_point().encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        )

        # Draw text labels near the points, and highlight based on selection
        text = line.mark_text(align='center', dx=20, dy=-20).encode(
            text=alt.condition(hover, 'CO2:Q', alt.value(' '))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(pred.astype(str)).mark_rule(color='gray').encode(
            x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
        ).transform_filter(
            hover
        )
        # Put the five layers into a chart and bind the data
        compare = alt.layer(
            line, selectors, points, rules, text
        ).properties(
            title="Actual vs. Predicted CO2 Concentration",
            height=400
        ).interactive()
        st.altair_chart(compare, use_container_width=True)

#         st.write("""
#         Next, we can do an out-of-time cross validation by taking few steps back in time and forecast into the future to as many steps you took back, then compare the forecast against the actuals.
#         Before that, let's split the dataset into training data and testing data with ratio of 75%:25% based on time frequency of series, because the order sequence of the time series should be intact in order to use it for forecasting.
#         """)
#         st.code("""
# train_end = int(round((len(co2["CO2_filled"])*0.75),0))
# train = co2["CO2_filled"][:train_end]
# test = co2["CO2_filled"][train_end:]
#         """)
#         train_end = int(round((len(co2[select_column]) * 0.75), 0))
#         train = co2["CO2_filled"][:train_end]
#         test = co2["CO2_filled"][train_end:]
#         st.write("""
#         Create ARIMA model using train data and fit the model. The orders (p, d, and q) are adjusted into 3, 2, 2 respectively to minimized the residuals.
#         """)
#         st.code("""
# train_model = ARIMA(train, order = (3,2,2))
# train_fit = train_model.fit()
# print(train_fit.summary())
#         """)
#         train_model = ARIMA(train.astype(float), order=(3,2,3))
#         train_fit = train_model.fit()
#         st.write(train_fit.summary())
#
#         test_pred = train_fit.forecast(len(test), alpha=0.05)
#
#         p = plt.figure(figsize=(10, 10), dpi=100)
#         ax = p.add_subplot(1, 1, 1)
#         ax.plot(train, label='Training', color='tab:blue')
#         ax.plot(test, label='Test Actual', color='tab:orange')
#         ax.plot(test_pred, label='Test Forecast', color='tab:green')
#
#         ax.set_title('Forecast vs Actuals')
#         ax.legend(loc='upper left', fontsize=8)
#         st.pyplot(p)

        st.subheader("5. Automatic ARIMA and SARIMA Forecasting")
        st.write("***Note :*** *use ARIMA if the selected column is seasonally adjusted, else use SARIMA*")
        method = st.selectbox("Choose Model", ["ARIMA", "SARIMA"])
        p1, p2 = st.columns(2)
        sp = p1.number_input("Start p", min_value=0, max_value=len(co2[select_column]) - 1, value=1, step=1, format="%d")
        sq = p1.number_input("Start q", min_value=0, max_value=len(co2[select_column]) - 1, value=1, step=1, format="%d")
        mp = p2.number_input("Max p", min_value=0, max_value=len(co2[select_column]) - 1, value=5, step=1, format="%d")
        mq = p2.number_input("Max q", min_value=0, max_value=len(co2[select_column]) - 1, value=5,step=1, format="%d")

        if sp > mp:
            st.error("Start p must be less than max p")
        elif sq > mq:
            st.error("Start q must be less than max q")
        else:
            if method == "ARIMA":
                st.write("""
                    ARIMA model has order p for AR model, order q for MA model, and order d for differencing. 
                    To determine their values, we can use ACF plot and PACF plot and choose the lags that are significant, then try to implement the order combinations on ARIMA model.
                    Python provides method called 'auto_arima' that can automatically identify the optimal order for the series.
                    """)
                arima_model = pm.auto_arima(co2[select_column], start_p=int(sp), start_q=int(sq),
                                                 test='adf', max_p=int(mp), max_q=int(mq),
                                                 m=12, d=None, seasonal=False,
                                                 start_P=0, D=0, trace=True,
                                                 error_action='ignore', suppress_warnings=True, stepwise=True)
                st.write(arima_model.summary())
                st.info("Best ARIMA Model : {}".format(arima_model))

                st.write("#### Residual Plot")
                st.pyplot(arima_model.plot_diagnostics(figsize=(10,8)))
                st.write("""
                - The **"Standardized residual"** plot shows that the residual seems to fluctuate around a mean of zero and have a uniform variance
                - The **"Histogram plus estimated density"** plot suggest normal distribution with mean zero
                - The **"Normal Q-Q"** plot shows that most of the dots fall nearly around the red line
                - The **"Correlogram"** plot or ACF plot shows that the autocorrelation of the residuals seems to have a pattern 
                """)

                st.write("#### Forecast Result")
                cl, cl1, cl2, cl3, cl4 = st.columns([3, 1, 1, 1, 1])
                periods = cl.number_input("Periods to be forcasted (months)", min_value=0, max_value=len(co2[select_column]) - 1, value=25, step=1, format="%d")
                lc_act_arima = cl1.color_picker("Actual color", value="#1f77b4")
                lc_pred_arima = cl2.color_picker("Forecast color", value="#228c22")
                pc_arima = cl3.color_picker("Point color", value="#d62728", key="pc_arima")
                rc_arima = cl4.color_picker("Rule color", value="#7f7f7f", key="rc_arima")

                combine = combine_data(co2, select_column, arima_model, periods)
                alt_chart = forecast_plot(combine, select_column, lc_act_arima, lc_pred_arima, rc_arima, pc_arima)
                st.altair_chart(alt_chart.properties(title="ARIMA - Final Forecast of CO2 Concentration"), use_container_width=True)

            elif method == "SARIMA":
                start_P = p1.number_input("Start P (seasonal)", min_value=0, max_value=len(co2[select_column]) - 1, value=0, step=1, format="%d")
                season_d = p2.number_input("Seasonal Differencing", min_value=0, max_value=len(co2[select_column]) - 1, value=1, step=1, format="%d")
                st.write("SARIMA model is the same with ARIMA except this model supports seasonality for the series. Therefore, there are other order such as P, D, and Q.")
                sarima_model = pm.auto_arima(co2[select_column],
                                         start_p=int(sp), start_q=int(sq), test='adf',
                                         max_p=int(mp), max_q=int(mq), m=12,
                                         start_P=int(start_P), seasonal=True,
                                         d=None, D=int(season_d), trace=True,
                                         error_action='ignore', suppress_warnings=True, stepwise=True)
                st.write(sarima_model.summary())
                st.info("Best SARIMA Model : {}".format(sarima_model))

                st.write("#### Residual Plot")
                st.pyplot(sarima_model.plot_diagnostics())
                st.write("""
                - The **"Standardized residual"** plot shows that the residual seems to fluctuate around a mean of zero and have a uniform variance
                - The **"Histogram plus estimated density"** plot suggest normal distribution with mean zero
                - The **"Normal Q-Q"** plot shows that most of the dots fall perfectly around the red line
                - The **"Correlogram"** plot or ACF plot shows that the autocorrelation of the residuals seems not having any patterns
                """)

                st.write("#### Forecast Result")
                cl, cl1, cl2, cl3, cl4 = st.columns([3, 1, 1, 1, 1])
                periods = cl.number_input("Periods to be forcasted (months)", min_value=0,
                                          max_value=len(co2[select_column]) - 1, value=25, step=1, format="%d")
                lc_act_sarima = cl1.color_picker("Actual color", value="#1f77b4")
                lc_pred_sarima = cl2.color_picker("Forecast color", value="#228c22")
                pc_sarima = cl3.color_picker("Point color", value="#d62728", key="pc_sarima")
                rc_sarima = cl4.color_picker("Rule color", value="#7f7f7f", key="rc_sarima")

                combine = combine_data(co2, select_column, sarima_model, periods)
                alt_chart = forecast_plot(combine, select_column, lc_act_sarima, lc_pred_sarima, rc_sarima, pc_sarima)
                st.altair_chart(alt_chart.properties(title="SARIMA - Final Forecast of CO2 Concentration"), use_container_width=True)

with st.expander("References", expanded = False):
    st.write("""
        <ul>
            <li><a href='https://scrippsco2.ucsd.edu/data/atmospheric_co2/primary_mlo_co2_record.html' target='_blank'>Scripps CO2 Website</a></li>
            <li><a href='https://streamlit.io/' target='_blank'>Streamlit Website</a></li>
            <li><a href='https://docs.streamlit.io/' target='_blank'>Streamlit Documentation</a></li>
            <li><a href='https://www.machinelearningplus.com/time-series/time-series-analysis-python/' target='_blank'>Time Series Analysis Guide</a></li>
            <li><a href='https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/' target='_blank'>Time Series Analysis Guide 2</a></li>
            <li><a href='https://machinelearningmastery.com/difference-time-series-dataset-python/' target='_blank'>Differencing Time Series Data</a></li>
            <li><a href='https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/' target='_blank'>ARIMA & SARIMA Model</a></li>
            
        </ul>
    """, unsafe_allow_html=True)


