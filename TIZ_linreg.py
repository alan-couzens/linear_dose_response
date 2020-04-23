import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

#Input variables time in zone (minutes) for Z1,Z2,Z3,Z4,Z5 per month
#Note: This could also be done from a csv file (e.g. from Training Peaks) using pd.read_csv

def get_block_end(start):
	end = start + timedelta(days=28)

def sum_TIZ_between_2_dates(data, col, date1, date2):
	date1 = pd.to_datetime(date1)
	date2 = pd.to_datetime(date2)
	total = data.loc[(data.WorkoutDay >= date1) & (data.WorkoutDay <= date2), col].sum()
	return total

def get_input_data_from_TrainingPeaks(file, block_length_days):
	data = pd.read_csv(file)
	data['WorkoutDay'] = pd.to_datetime(data['WorkoutDay'])
	start = data['WorkoutDay'].values[0]
	end = data['WorkoutDay'].values[-1]
	days = (end - start)/np.timedelta64(1,'D')
	blocks = int(days/block_length_days)
	block_dates = []
	for i in range(blocks+1):
		new_date = start + np.timedelta64(i*block_length_days, 'D')
		block_dates.append(np.datetime64(new_date, 'D'))
	intensities = ['HRZone1Minutes', 'HRZone2Minutes', 'HRZone3Minutes', 'HRZone4Minutes', 'HRZone5Minutes', 'HRZone6Minutes', 'HRZone7Minutes']
	input_data = []
	for i in range(len(block_dates)):
		try:
			block_TIZ = []
			for intensity in intensities:
				total_TIZ = sum_TIZ_between_2_dates(data, intensity, block_dates[i], block_dates[i+1])
				block_TIZ.append(total_TIZ)
			input_data.append(block_TIZ)
		except IndexError:
			pass
	return input_data

#Input Data for the model - will look for a workouts.csv file from Training Peaks. If not found, resorts to your manual entry of TIZ by block
try: 
	time_in_zone_by_month = get_input_data_from_TrainingPeaks('workouts.csv')
except:
	time_in_zone_by_month=[
	[1205,902,330,48,20],
	[1303,1021,371,69,19],
	[1370,1311,380,53,24],
	[1389,1330,391,118,18],
	[1333,1291,458,109,28],
	]

#Output variable - tested FTP each month (replace with your own data)
FTP_by_month = [248, 279, 295, 310, 303]

X = time_in_zone_by_month
y = FTP_by_month

model = LinearRegression().fit(X,y)

"""Print out model weights for each zone so that you can see how strongly they are
weighted/how much each minute in each zone is worth. e.g. the first coefficient is the
weight for Zone 1, the second is the weight for Zone 2 etc."""
print(model.coef_)
print(model.intercept_)



