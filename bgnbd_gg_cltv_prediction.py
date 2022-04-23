import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from sqlalchemy import create_engine
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Reading data from excel
df_ = pd.read_excel("HAFTA_03/Ders Notları/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

# Our data so big that why i am taking copy of my data
df = df_.copy()

# When we check our data we can see multiple invoice for 1 order.
df.head()

#DB connection

#credentials.
creds = {'user': '..........',
         'passwd': '..........',
         'host': '.........,
         'port':.......,
         'db': '........'}

#MySQL connection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

#Sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

# conn.close()

###############################################################
# Data Understanding
###############################################################
df.shape
#(541910, 8)
#Observational Unit: 541910
#Variable: 8

#I want to check my data if there is a outlier problem etc.
df.describe().T

###############################################################
# Data Preparation
###############################################################

df.dropna(inplace=True)

#C returned product so i did not take them
df = df[~df["Invoice"].str.contains("C", na=False)]

#POST is postage cost so i deleted it is not a product
df = df[~df["StockCode"].str.contains("POST", na=False)]

#There is no total prices in df so i created new feature named ‘TotalPrice’
df["TotalPrice"] = df["Quantity"] * df["Price"]

#We have outlier problems so i replaced values with thresholds
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

#customer's latest purchase date
df["InvoiceDate"].max()
#Timestamp('2011-12-09 12:50:00')

#Taking measurment date as 2 days after from latest purchase date
today_date = dt.datetime(2011, 12, 11)

###############################################################
# Preparation of CLTV Data Structure
###############################################################
#Calculate the monetary value for the Gamma-Gamma Model.

# recency: Calculation of the time elapsed since the last purchase for each customer(Weakly).
# T: how long time passed until the first purchase date to analysis date (Weakly).Age of customer.
# frequency: Total number of repeat purchases (frequency>1) Customer should purchased at least
# 2 times to consider not churned customers
# monetary_value: Average earnings for per purchase

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

#We used it because there is a hierarchical index problem (the reason is 2 lambdas in InvoiceDate)
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']


#Average earnings for per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# Customer should purchased at least 2 times
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

#Average earnings for per purchase should be bigger tha 0
cltv_df = cltv_df[cltv_df["monetary"] > 0]

#Making data weekly
cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7


##############################################################
# Setting BG-NBD Model :Modelling buying and drop out process.(excepted number of transiction)
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

##############################################################
# Setting GAMMA-GAMMA Model :Excepted average profit
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


#10 highest revenue customers
cltv_df.head(10)

#Lets make a 6-month CLTV prediction
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  #montly
                                   freq="W",  # T freq..(weakly)
                                   discount_rate=0.01)
#Fixing index problem
cltv = cltv.reset_index()

#Lets have a look 50 people with the highest CLTV score
cltv.sort_values(by="clv", ascending=False).head(50)

#10 most purchases in the next 6 months
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

#For better understanding i like to make min-max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.sort_values(by="scaled_clv", ascending=False).head()

#Lets make a 1-month CLTV prediction
cltv_1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

#Lets make a 1-year CLTV prediction
cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)
#I want to see if our custemers life time value increased or decreased by time.Lets check difference!
cltv_1 = cltv_1.reset_index()
cltv_1=cltv_1.sort_values(by="clv", ascending=False)
cltv_1=cltv_1[["Customer ID","clv"]]
cltv_1.head(10)

cltv_12 = cltv_12.reset_index()
cltv_12=cltv_12.sort_values(by="clv", ascending=False)
cltv_12=cltv_12[["Customer ID","clv"]]
cltv_12.head(10)

#1 MONTH
# Customer ID        clv
# 1963   16446.0000 38990.5475
# 1115   14646.0000 19812.0320
# 2753   18102.0000 18013.6521
# 2450   17450.0000 13598.6101
# 836    14096.0000 12027.8612
# 35     12415.0000  9783.1454
# 1250   14911.0000  9595.1771
# 867    14156.0000  8051.7544
# 1747   16000.0000  7805.5843
# 2479   17511.0000  6441.8559

#12 MONTH
#    Customer ID         clv
# 0   16446.0000 420893.7716
# 1   14646.0000 217588.1766
# 2   18102.0000 197986.7058
# 3   17450.0000 149450.4733
# 4   14096.0000 128011.4781
# 5   12415.0000 107271.1929
# 6   14911.0000 105515.6695
# 7   14156.0000  88515.4097
# 8   16000.0000  79643.5944
# 9   17511.0000  70811.9218

#As you can see CLTV score increased for all top 10 customer.No change in rankings.
#In other words, the lifetime values of customers are increasing day by day.

##############################################################
# Segmentation According to CLTV
##############################################################

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()

cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)

# Segmentleri betimleyelim:
cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})

cltv_final.head()

#    Customer ID  recency       T  frequency  monetary       clv  scaled_clv segment
# 0   12347.0000  52.1429 52.5714          7  615.7143 2200.3501      0.0099       A
# 1   12348.0000  40.2857 51.2857          4  359.3100  853.5763      0.0038       C
# 2   12352.0000  37.1429 42.4286          8  278.2550 1276.6212      0.0057       B
# 3   12356.0000  43.1429 46.5714          3  829.1433 1867.3840      0.0084       A
# 4   12358.0000  21.2857 21.5714          2  464.0300 1509.9655      0.0068       B

#When we check the reasult as you can see first customer has less moetary than third but frequency is 7, betteer than
#third one.So cltv prediction helping us to combining all variables and gives total score.

#Lets choose 2 gorup and make some comment
cltv[cltv_final["segment"]=="A"].sort_values(by="clv",ascending=False).head()
#Group A has best CLTV value.We should make something to make them spend and buy more.
#We can apply apply up-selling,cross selling strategy.We can make product reccomendation.


cltv[cltv_final["segment"]=="D"].sort_values(by="clv",ascending=False).head()
#They have lowest CLTV score.We can send some notifications, mails or messages like we missed you.
# Making some campaign, special offer etc. can make them spend and stay with our company.

# Sending final table to database
cltv_final["CustomerID"] = cltv_final["CustomerID"].astype(int)
cltv_final.to_sql(name='ZEYNEP_BAYKAN', con=conn, if_exists='replace', index=False)
