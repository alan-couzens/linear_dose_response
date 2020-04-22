import pandas as pd
from sklearn.linear_model import LinearRegression

#Input variables time in zone (minutes) for Z1,Z2,Z3,Z4,Z5 per month
#Note: This could also be done from a csv file (e.g. from Training Peaks) using pd.read_csv
#Sample data
time_in_zone_by_month=[
[1205,902,330,48,20],
[1303,1021,371,69,19],
[1370,1311,380,53,24],
[1389,1330,391,118,18],
[1333,1291,458,109,28],
]

#Output variable - tested FTP each month (sample data)
FTP_by_month = [248, 279, 295, 310, 303]

X = time_in_zone_by_month
y = FTP_by_month

model = LinearRegression().fit(X,y)

"""Print out model weights for each zone so that you can see how strongly they are
weighted/how much each minute in each zone is worth. e.g. the first coefficient is the
weight for Zone 1, the second is the weight for Zone 2 etc."""
print(model.coef_)
print(model.intercept_)



