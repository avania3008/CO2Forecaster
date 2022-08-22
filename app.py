from datetime import date, datetime as dt
import os
import tempfile
import base64
import re
import uuid
from io import StringIO
import streamlit as st
from csv import reader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
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

#configs
temp_path = tempfile.gettempdir()
st.set_page_config(page_title="Atmospheric CO2", layout="wide")

#styling
css = '''
<style>
    p {
        text-align: justify;
    }

    div.row-widget.stRadio > div {
        flex-direction:row;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

#functions
@st.experimental_memo(show_spinner=False, suppress_st_warning=True)
def get_data(file, type):
    header = ["Yr", "Mn", "Date1", "Date2", "CO2", "seasonally_adjusted", "fit",
              "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]

    if type == "upload":
        with StringIO(file.getvalue().decode("utf-8")) as f:
            rows = list(reader(f))
            selected_rows = rows[57:]
        df = pd.DataFrame(selected_rows, columns = header)
    elif type == "in-built":
        with open(file,"r") as f:
            selected_rows = f.readlines()[57:]
        data_list_of_lists = [data.split(", ") for data in selected_rows]
        df = pd.DataFrame(data_list_of_lists, columns = header)
    
    df = df.apply(pd.to_numeric)
    return df        

@st.experimental_memo(show_spinner=False, suppress_st_warning=True)
def csv_dwl_button(df, label, file_name):
    file_path = f"{temp_path}/{file_name}.csv"
    df.to_csv(file_path, index = False)
    with open(file_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    file_size = os.path.getsize(file_path)
    file_size_kb = round(file_size/(1024), 2)
    if file_size_kb > 1000:
        file_size_mb = round(file_size/(1024*1024), 2)
        dwl_label = f"{label} ({file_size_mb} MB)"
    else:
        dwl_label = f"{label} ({file_size_kb} KB)"

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """
    dl_link = custom_css + f'<a download="{file_name}.csv" id="{button_id}" href="data:text/csv;base64,{b64}">{dwl_label}</a><br></br>'
    return st.markdown(dl_link, unsafe_allow_html=True)

@st.experimental_memo(show_spinner=False, suppress_st_warning=True)
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

@st.experimental_memo(show_spinner=False, suppress_st_warning=True)
def combine_data(actual, column, model, periods):
    fitted = model.predict(n_periods=int(periods), return_conf_int=False)
    index_of_fc = pd.date_range(actual[column].index[0], periods=int(periods), freq='MS')
    fitted_series = pd.DataFrame(fitted, index=index_of_fc)
    fitted_series.index.name = "Time"
    fitted_series.rename(columns={0: "Forecast"}, inplace=True)

    combine = pd.concat([actual, fitted_series], axis=0)
    combine_noidx = combine.reset_index()
    combine_noidx["Month"] = combine_noidx["Time"].apply(lambda x: x.strftime("%b"))
    combine_noidx["Year"] = combine_noidx["Time"].apply(lambda x: x.strftime("%Y"))
    return combine_noidx

@st.experimental_memo(show_spinner=False, suppress_st_warning=True)
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

#sidebar
st.sidebar.image('img/scrippslogo-removebg.png') 
avl_file = st.sidebar.selectbox("Select Avaliable Data", ["Jan. 1958 - Sept. 2021","Jan. 1958 - Jul. 2022"], key = "avl_file")
st.sidebar.markdown("<p style='text-align:center;'><strong>OR</strong></p>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload Atmospheric CO2 Data (CSV)", type = "csv", key = "uploaded_file")
if st.session_state["uploaded_file"]:
    st.sidebar.success("New file successfully uploaded!")
st.sidebar.markdown('''
<p><strong>Info :</strong> You can download the Atmospheric CO2 newest data from 
<a href='https://scrippsco2.ucsd.edu/data/atmospheric_co2/primary_mlo_co2_record.html' target='_blank'>here</a><p>
''', unsafe_allow_html=True)

if "curr_file" not in st.session_state:
    st.session_state["curr_file"] = None

if st.session_state["uploaded_file"]:
    st.session_state["curr_file"] = st.session_state["uploaded_file"].name
else :
    st.session_state["curr_file"] = st.session_state["avl_file"]

# if st.session_state["uploaded_file"]:
#     st.sidebar.markdown("""
#             <strong>Uploaded File Details</strong>
#             <table>
#                 <tr>
#                     <th>Name</th>
#                     <td><i>{a}</i></td>
#                 </tr>
#                 <tr>
#                     <th>Type</th>
#                     <td><i>{b}</i></td>
#                 </tr>
#                 <tr>
#                     <th>Size</th>
#                     <td><i>{c} Bytes</i></td>
#                 </tr>
#             </table>
#             <br>
#             """.format(a=uploaded_file.name, b=uploaded_file.type, c=uploaded_file.size), unsafe_allow_html=True)
        
# main page
st.header("Monthly Forecaster: Atmospheric CO2 of Mauna Loa Observatory, Hawaii")
st.markdown("<p><strong>Created by </strong><a href='https://github.com/avania3008' target='_blank'>Aurellia Vania Yosephine Budiman</a></p>", unsafe_allow_html=True)

with st.expander("Show Dataset", expanded = True):
    st.info("**Used data file** : {}".format(st.session_state["curr_file"]))
    if ("used_df" not in st.session_state) or (not st.session_state["uploaded_file"]):
        if st.session_state["avl_file"] == "Jan. 1958 - Sept. 2021":
            st.session_state["used_df"] = get_data("data/monthly_in_situ_co2_mlo_sept_2021.csv","in-built")
        elif st.session_state["avl_file"] == "Jan. 1958 - Jul. 2022":
            st.session_state["used_df"] = get_data("data/monthly_in_situ_co2_mlo_july_2022.csv","in-built")
    else:
        st.session_state["used_df"] = get_data(st.session_state["uploaded_file"],"upload")

    st.write("The raw CSV file downloaded from Scripps CO2 website contains some data informations and unstructured table headers, so I decided to format the file into a structured table that only includes the main data. Here is the formatted table from the currently used data file :")
    AgGrid(st.session_state["used_df"])
    st.write("**Note** : *Missing values are denoted by -99.99*")
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
    """)

with st.expander("Raw Dataset Visualization", expanded = False):
    st.write("This part shows the line chart of the CO2 concentrations from certain range of date")
    period = st.selectbox('Select period', ["Monthly", "Yearly"], key="period")
    c1, c2 = st.columns(2)
    smoothed = c1.radio("Smoothed", ["Yes", "No"], key="smoothed")
    seasonally = c2.radio("Seasonally Adjusted", ["Yes", "No"], key="seasonally")

    # preprocess dataframe
    # get min, median, max of year
    min_year = min(st.session_state["used_df"]["Yr"])
    max_year = max(st.session_state["used_df"]["Yr"])
    mid_year = int(np.quantile(st.session_state["used_df"]["Yr"], 0.5))

    # convert year & month (int) to string
    st.session_state["used_df"]["Yr"] = st.session_state["used_df"]["Yr"].apply(lambda x: str(x))
    st.session_state["used_df"]["Mn"] = st.session_state["used_df"]["Mn"].apply(lambda x: str(x).strip())

    # rename column
    st.session_state["used_df"] = st.session_state["used_df"].rename(columns={"Yr": "Year"})
    # convert month number to month name
    month_dict = {"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr",
                  "5": "May", "6": "Jun", "7": "Jul", "8": "Aug",
                  "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
    st.session_state["used_df"]["Month"] = st.session_state["used_df"]["Mn"].map(month_dict)
    # concat month name and year
    st.session_state["used_df"]["Time"] = st.session_state["used_df"]["Mn"] + "-" + st.session_state["used_df"]["Year"]
    # convert time (string type) to datetime type
    st.session_state["used_df"]["Time"] = st.session_state["used_df"]["Time"].apply(lambda x: dt.strptime(x, "%m-%Y"))

    # choose column to visualize by user input (radio button)
    if "curr_col" not in st.session_state:
        st.session_state["curr_col"] = None
    if st.session_state["smoothed"] == "Yes" and st.session_state["seasonally"] == "Yes":
        st.session_state["curr_col"] = "seasonally_adjusted_fit"
        #null_row = df[df[col] == -99.99].index
        #df[col] = df[col].replace([-99.99],df["seasonally_adjusted_filled"][null_row])
    elif st.session_state["smoothed"] == "Yes" and st.session_state["seasonally"] == "No":
        st.session_state["curr_col"] = "fit"
        #null_row = df[df[col] == -99.99].index
        #df[col] = df[col].replace([-99.99], df["CO2_filled"][null_row])
    elif st.session_state["smoothed"] == "No" and st.session_state["seasonally"] == "Yes":
        st.session_state["curr_col"] = "seasonally_adjusted_filled"
    else:
        st.session_state["curr_col"] = "CO2_filled"

    # show used column for the chart
    c3, c4, c5, c6 = st.columns([10, 1, 1, 1])
    c3.info("**Used column** : {}".format(st.session_state["curr_col"]))
    line_color = c4.color_picker("Line color", value = "#1f77b4", key="line_color")
    point_color = c5.color_picker("Point color", value = "#d62728", key="point_color")
    rule_color = c6.color_picker("Rule color", value = "#7f7f7f", key="rule_color")
        
    if st.session_state["period"] == "Monthly" :
        start_month = c1.selectbox('Start Month', ["January", "February", "March", "April", "May", "June", "July"
                                                    , "August", "September", "October", "November", "December"], key="start_month")
        
        end_month = c2.selectbox('End Month', ["January", "February", "March", "April", "May", "June", "July"
                                                , "August", "September", "October", "November", "December"], key="end_month") 
        start_year = c1.slider('Start Year', min_value=min_year, max_value=max_year, value=mid_year, key="start_year")
        end_year = c2.slider('End Year', min_value=min_year, max_value=max_year, value=max_year, key="end_year")
        std = dt.strptime((str(start_month)+"-"+str(start_year)), "%B-%Y")
        end = dt.strptime((str(end_month)+"-"+str(end_year)), "%B-%Y")
        if std > end:
            st.error("The start time mustn't be after the end time")
        else:
            time_ranged = st.session_state["used_df"].loc[((st.session_state["used_df"]['Time'] >= std) & (st.session_state["used_df"]['Time'] <= end))]
            data = time_ranged[time_ranged[st.session_state["curr_col"]] != -99.99]
            monthly = line_chart(data,st.session_state["curr_col"],True,st.session_state["line_color"],st.session_state["point_color"],st.session_state["rule_color"])
            st.altair_chart(monthly, use_container_width=True)
    else :
        start_year = c1.slider('Start Year', min_value=min_year, max_value=max_year, value=mid_year, key="start_year")
        end_year = c2.slider('End Year', min_value=min_year, max_value=max_year, value=max_year, key="end_year")
        #convert date to datetime
        std = dt.strptime(str(start_year), "%Y")
        end = dt.strptime(str(end_year), "%Y")
        if std > end:
            st.error("The start time mustn't be after the end time")
        else:
            time_ranged = st.session_state["used_df"].loc[((st.session_state["used_df"]['Time'] >= std) & (st.session_state["used_df"]['Time'] <= end))]
            data = time_ranged[time_ranged[st.session_state["curr_col"]] != -99.99]
            yearly = line_chart(data, st.session_state["curr_col"], False,st.session_state["line_color"],st.session_state["point_color"],st.session_state["rule_color"])
            st.altair_chart(yearly, use_container_width=True)

with st.expander("Analysis Results", expanded = False):
    st.subheader("1. Data Preparation")
    st.markdown("""
    <div>
        <p>Using the formatted data frame, the needed steps are :</p>
        <ul>
            <li>Drop rows that have missing values (value = -99.99) on the selected column</li>
            <li>Create a new column named 'Time' that was converted from 'Mn' column and 'Yr' column</li>
            <li>Subset the columns into only 'Time' and a column to be analyzed (one column from column 7-10)</li>
            <li>Set 'Time' column as index</li>
        </ul>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    select_column = st.selectbox('Select the column to be analyzed', ['CO2_filled', 'seasonally_adjusted_filled', 'fit', 'seasonally_adjusted_fit'], key="select_column")
    if "co2" not in st.session_state:
        st.session_state["co2"] = None
 
    # select non missing values
    st.session_state["co2"] = st.session_state["used_df"][st.session_state["used_df"][st.session_state["select_column"]] != -99.99]
    # columns selection
    st.session_state["co2"] = st.session_state["co2"][["Time", st.session_state["select_column"]]]
    # set index
    st.session_state["co2"].set_index("Time", inplace = True)

    col1, col2 = st.columns(2)
    col1.code("co2.head()")
    col1.dataframe(st.session_state["co2"].head())
    col2.code("co2.tail()")
    col2.dataframe(st.session_state["co2"].tail())
    
    st.subheader("2. Check for Stationarity and Seasonality")
    st.markdown("""
    <p>
    Next before building the model, we need to check the time series stationarity and seasonality.
    One of the way to do that is by decomposing the time series into its components, such as base level, trend, seasonality, and error.
    Depending on the nature of the trend and seasonality, a time series can be modeled as an additive or multiplicative :</p>

    <ul>
        <li><strong>Additive Time Series</strong> : Base Level + Trend + Seasonality + Error</li>
        <li><strong>Multiplicative Time Series</strong> : Base Level * Trend * Seasonality * Error</li>
    </ul> 
    """, unsafe_allow_html=True)
    st.code("""
    # decomposing time series to its components
    # Additive Decomposition
    result_add = seasonal_decompose(co2['{x}'], model='additive', extrapolate_trend='freq')

    # Multiplicative Decomposition 
    result_mul = seasonal_decompose(co2['{x}'], model='multiplicative', extrapolate_trend='freq')
    """.format(x=st.session_state["select_column"]))
    result_add = seasonal_decompose(st.session_state["co2"][st.session_state["select_column"]], model='additive', extrapolate_trend='freq')
    result_mul = seasonal_decompose(st.session_state["co2"][st.session_state["select_column"]], model='multiplicative', extrapolate_trend='freq')
    sd_add, sd_mul = st.columns(2)

    sd_add.markdown("<p style='text-align:center'><strong>Additive Decomposition</strong></p>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(3, 1, figsize=(10,12))
    ax1[0].title.set_text("Trend")
    ax1[0].plot(result_add.trend)
    ax1[1].title.set_text("Seasonality")
    ax1[1].plot(result_add.seasonal)
    ax1[2].title.set_text("Residual")
    ax1[2].plot(result_add.resid)
    sd_add.pyplot(fig1)

    sd_mul.markdown("<p style='text-align:center'><strong>Multiplicative Decomposition</strong></p>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(3, 1, figsize=(10,12))
    ax2[0].title.set_text("Trend")
    ax2[0].plot(result_mul.trend)
    ax2[1].title.set_text("Seasonality")
    ax2[1].plot(result_mul.seasonal)
    ax2[2].title.set_text("Residual")
    ax2[2].plot(result_mul.resid)
    sd_mul.pyplot(fig2)

    st.markdown("""
    <p>
    From the results of additive decomposition and multiplicative decomposition, we can see the dataset's trend pattern and seasonal pattern. 
    If the dataset have trend pattern, seasonal pattern, or both pattern, it means that the data is non-stationary.
    To be more certain, let's do the stationarity check using Augmented Dickey Fuller (ADF) test or Unit Root Test.</p>
        
    **ADF Test**\n
    *H0 : The time series is non-stationary*\n
    *H1 : The time series is stationary*\n
    """, unsafe_allow_html=True)

    st.code("adf_test = adfuller(co2['{}'].values, autolag='AIC')".format(st.session_state["select_column"]))

    adf_test = adfuller(st.session_state["co2"][st.session_state["select_column"]].values, autolag='AIC')
    adf, pval = st.columns(2)
    adf.metric(label="ADF Statistic", value=adf_test[0])
    pval.metric(label="P-Value", value=adf_test[1])
    
    res = "Reject" if (adf_test[1] < 0.05) else "Fail to reject"
    st.info(f"**Conclusion** : {res} the null hypothesis")
    st.write("""
    Using above results, if the p-value test is less than the significance level (0.05), reject the null hypothesis or the time series is stationary. 
    Otherwise, fail to reject the null hyphotesis, which means the time series is non-stationary. 
    To make non-stationary into stationary for effective time series modeling, one of the popular methods is by **differencing**.
    
    One of most popular models for time series forecasting is ARIMA (Auto Regressive Integrated Moving Average). Differencing can be done alongside we build the ARIMA model. 
    Also, we need to notice that ARIMA model doesn't support seasonality, therefore we can use SARIMA (Seasonal ARIMA) as an alternative for the seasonal time series.
    """)

    st.subheader("3. ACF Plot and PACF Plot")
    st.write("""
    Before building the appropriate model, let's see its ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots.
    """)
    Acf, Pacf = st.columns(2)
    Acf.pyplot(plot_acf(st.session_state["co2"][st.session_state["select_column"]], title = "Autocorrelation of "+ str(st.session_state["select_column"])))
    Pacf.pyplot(plot_pacf(st.session_state["co2"][st.session_state["select_column"]], title="Partial Autocorrelation of " + str(st.session_state["select_column"])))
    st.write("""
    If the autocorrelations are positive for many number of lags (10 or more) and goes into the far negative zone slowly, then the series needs further **differencing**. 
    It indicates that the series tend to have positive trend pattern on it. On the other hand, if the autocorrelation goes into the far negative zone very quickly, then the series is probably over-differenced.
    """)

#     st.subheader("Building ARIMA Model Manually")
#     st.write("""
#     ***Order Combinations of ARIMA(p,d,q):***
#     - ARIMA(1,1,1)
#     - ARIMA(1,1,2)
#     - ARIMA(1,2,1)
#     - ARIMA(2,1,1)
#     - ARIMA(2,1,2)
    
#     Using above combinations, the best model is the fourth model with the smallest AIC = 1656.508, also all of the variables are significant with p-value less than significance level 0.05. Here is the summary :
#     """)
#     st.code("""
# model4 = ARIMA(co2["CO2_filled"], order=(2,1,1), freq=co2.index.inferred_freq)
# model4_fit = model4.fit()
# print(model4_fit.summary())
#         """)
#     model4 = ARIMA(co2[select_column].astype(float), order=(2, 1, 1), freq=co2.index.inferred_freq)
#     model4_fit = model4.fit()
#     st.write(model4_fit.summary())
#     st.write("""
#     With the above model, we can plot the residuals to ensure that there aren't any pattern (it means having constant mean and variance) and also predict the fit value of CO2 concentration.
#     """)
#     st.write("#### Residual Plot")
#     residuals = pd.DataFrame(model4_fit.resid)
#     g = residuals.plot(title="Residuals", ylim=[-3.5, 3.5])
#     g.set_xlabel('Time')
#     g.set_ylabel('Residuals')
#     resid_plot = g.figure
#     st.pyplot(resid_plot)
#     st.code("""
# # Actual vs Fitted
# fitted = model4_fit.predict()
# predicted = co2.copy()
# predicted["Fitted"] = np.around(fitted.values, 2)
# predicted.rename(columns = {"CO2_filled": "Actual"}, inplace = True)
# predicted['Fitted'] = predicted['Fitted'].replace(0, np.nan)
# print(predicted.head())
#         """)
#     fitted = model4_fit.predict()
#     predicted = co2.copy()
#     predicted["Fitted"] = np.around(fitted.values, 2)
#     predicted.rename(columns={select_column: "Actual"}, inplace=True)
#     predicted['Fitted'] = predicted['Fitted'].replace(0, np.nan)
#     st.write(predicted.head())
#     st.write("The residual errors seem fine with near zero mean and uniform variance, so let’s plot the actuals against the fitted values :")
#     pred = predicted.reset_index().melt('Time', var_name='Type', value_name='CO2').dropna()
#     hover = alt.selection_single(
#         encodings=["x"],
#         nearest=True,
#         on="mouseover",
#         empty="none",
#         clear="mouseout"
#     )
#     # The basic line
#     line = alt.Chart(pred.astype(str)).mark_line(interpolate='basis').encode(
#         x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
#         y=alt.Y('CO2:Q', axis=alt.Axis(title='CO2 Concentration (ppm)')),
#         color='Type:N'
#     )
#     # Transparent selectors across the chart. This is what tells us
#     # the x-value of the cursor
#     selectors = alt.Chart(pred.astype(str)).mark_point().encode(
#         x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
#         opacity=alt.value(0),
#         tooltip=[alt.Tooltip("Time:O", timeUnit="yearmonth", title="Time")]
#     ).add_selection(
#         hover
#     )
#     # Draw points on the line, and highlight based on selection
#     points = line.mark_point().encode(
#         opacity=alt.condition(hover, alt.value(1), alt.value(0))
#     )
#     # Draw text labels near the points, and highlight based on selection
#     text = line.mark_text(align='center', dx=20, dy=-20).encode(
#         text=alt.condition(hover, 'CO2:Q', alt.value(' '))
#     )
#     # Draw a rule at the location of the selection
#     rules = alt.Chart(pred.astype(str)).mark_rule(color='gray').encode(
#         x=alt.X("Time:O", timeUnit="yearmonth", axis=alt.Axis(format="%b-%Y"), title="Time"),
#     ).transform_filter(
#         hover
#     )
#     # Put the five layers into a chart and bind the data
#     compare = alt.layer(
#         line, selectors, points, rules, text
#     ).properties(
#         title="Actual vs. Predicted CO2 Concentration",
#         height=400
#     ).interactive()
#     st.altair_chart(compare, use_container_width=True)

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




    # st.subheader("4. Automatic ARIMA and SARIMA Forecasting")
    # st.write("""
    # ARIMA model has order p for AR model, order q for MA model, and order d for differencing. 
    # To determine their values, we can use ACF plot and PACF plot and choose the lags that are significant, then try to implement the order combinations on ARIMA model.
    # Auto-ARIMA method can automatically identify the optimal order for the series.
    # """)
    # st.write("SARIMA model is the same with ARIMA, except this model supports seasonality for the series. Therefore, there are other order such as P, D, and Q.")
    # st.write("**Note :** *Auto-ARIMA method is more suitable for the seasonally adjusted data, while Auto-SARIMA used for the not seasonally adjusted data*")
    # p1, p2 = st.columns(2)
    # sp = p1.number_input("Start p", min_value=0, max_value=len(st.session_state["co2"][st.session_state["select_column"]]) - 1, value=2, step=1, format="%d", key="sp")
    # sq = p1.number_input("Start q", min_value=0, max_value=len(st.session_state["co2"][st.session_state["select_column"]]) - 1, value=2, step=1, format="%d", key="sq")
    # mp = p2.number_input("Max p", min_value=0, max_value=len(st.session_state["co2"][st.session_state["select_column"]]) - 1, value=5, step=1, format="%d", key="mp")
    # mq = p2.number_input("Max q", min_value=0, max_value=len(st.session_state["co2"][st.session_state["select_column"]]) - 1, value=5,step=1, format="%d", key="mq")
    # if st.session_state["sp"] > st.session_state["mp"]:
    #     st.error("Start p must be less than max p")
    # elif st.session_state["sq"] > st.session_state["mq"]:
    #     st.error("Start q must be less than max q")
    # else:
    #     if ["best_model","model_method"] not in st.session_state:
    #         st.session_state["best_model"] = None
    #         st.session_state["model_method"] = None

    #     if "seasonally_adjusted" in st.session_state["select_column"]:
    #         st.session_state["model_method"] = "ARIMA" 

    #         @st.experimental_memo(show_spinner=False, suppress_st_warning=True)
    #         def create_arima_model():
    #             arima_model = pm.auto_arima(st.session_state["co2"][st.session_state["select_column"]], 
    #                                         start_p=int(st.session_state["sp"]), 
    #                                         start_q=int(st.session_state["sq"]),
    #                                         test='adf', 
    #                                         max_p=int(st.session_state["mp"]), 
    #                                         max_q=int(st.session_state["mq"]),
    #                                         m=12, d=None, seasonal=False,
    #                                         start_P=0, D=0, trace=True,
    #                                         error_action='ignore', suppress_warnings=True, stepwise=True)
    #             return arima_model
    #         st.session_state["best_model"] = create_arima_model()
    #     else :
    #         st.session_state["model_method"] = "SARIMA"
    #         start_P = p1.number_input("Start P (seasonal)", min_value=0, max_value=len(st.session_state["co2"][st.session_state["select_column"]]) - 1, value=0, step=1, format="%d", key="sP")
    #         season_d = p2.number_input("Seasonal Differencing", min_value=0, max_value=len(st.session_state["co2"][st.session_state["select_column"]]) - 1, value=1, step=1, format="%d", key="sd")
           
    #         @st.experimental_memo(show_spinner=False, suppress_st_warning=True)
    #         def create_sarima_model():
    #             sarima_model = pm.auto_arima(st.session_state["co2"][st.session_state["select_column"]],
    #                                         start_p=int(st.session_state["sp"]), 
    #                                         start_q=int(st.session_state["sq"]),
    #                                         test='adf', 
    #                                         max_p=int(st.session_state["mp"]), 
    #                                         max_q=int(st.session_state["mq"]),
    #                                         m=12, d=None, seasonal=True,
    #                                         start_P=int(st.session_state["sP"]),
    #                                         D=int(st.session_state["sd"]), trace=True,
    #                                         error_action='ignore', suppress_warnings=True, stepwise=True)
    #             return sarima_model
    #         st.session_state["best_model"] = create_sarima_model()
        
    #     # st.info("**Used method** : Auto-{}".format(st.session_state["model_method"]))
    #     # summary
    #     st.write("#### Result")
    #     # st.write(st.session_state["best_model"].summary())
    #     st.success("**Best {method} Model** : {model}".format(method = st.session_state["model_method"], model = st.session_state["best_model"]))

    #     # resid plot
    #     st.checkbox("Show residual plot", key="resid_plot")
    #     if st.session_state["resid_plot"]:
    #         st.write("#### Residual Plot")
    #         st.pyplot(st.session_state["best_model"].plot_diagnostics(figsize=(10,8)))
    #         # st.write("""
    #         # - The **"Standardized residual"** plot shows that the residual seems to fluctuate around a mean of zero and have a uniform variance
    #         # - The **"Histogram plus estimated density"** plot suggest normal distribution with mean zero
    #         # - The **"Normal Q-Q"** plot shows that most of the dots fall nearly around the red line
    #         # - The **"Correlogram"** plot or ACF plot shows that the autocorrelation of the residuals seems to have a pattern 
    #         # """)

    #     # forecast visualization
    #     st.write("#### Forecast Result")
    #     cl, cl1, cl2, cl3, cl4 = st.columns([3, 1, 1, 1, 1])
    #     periods = cl.number_input("Periods to be forecasted (months)", min_value=0, max_value=len(st.session_state["co2"][st.session_state["select_column"]]) - 1, value=25, step=1, format="%d", key="periods")
    #     lc_act_arima = cl1.color_picker("Actual color", value="#1f77b4", key="lc_act_arima")
    #     lc_pred_arima = cl2.color_picker("Forecast color", value="#228c22", key="lc_pred_arima")
    #     pc_arima = cl3.color_picker("Point color", value="#d62728", key="pc_arima")
    #     rc_arima = cl4.color_picker("Rule color", value="#7f7f7f", key="rc_arima")
    #     if "df_result" not in st.session_state:
    #         st.session_state["df_result"] = None

    #     st.session_state["df_result"] = combine_data(st.session_state["co2"], st.session_state["select_column"], st.session_state["best_model"], st.session_state["periods"])
    #     alt_chart = forecast_plot(st.session_state["df_result"], st.session_state["select_column"], st.session_state["lc_act_arima"], st.session_state["lc_pred_arima"], st.session_state["rc_arima"], st.session_state["pc_arima"])
    #     st.altair_chart(alt_chart.properties(title="{method} - Final Forecast of CO2 Concentration ({col})".format(method = st.session_state["model_method"], col = st.session_state["select_column"])), use_container_width=True)
    #     st.write("To save the forecast results, click the button below :")
    #     csv_dwl_button(st.session_state["df_result"], "Download Forecast Result", "mauna_loa_co2_forecast_results")   
        
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


