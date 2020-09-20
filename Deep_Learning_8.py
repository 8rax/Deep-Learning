#https://pythonprogramming.net/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/

import pandas as pd
import os

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3 
RATIO_TO_PREDICT = "LTC-USD"

#We load the data

#Printing the head
#print(df.head())

#First, we need to combine price and volume for each coin into a single featureset, then we want to take these featuresets and combine them into sequences of 60 of these featuresets. This will be our input.

#Okay, what about our output? Our targets? Well, we're trying to predict if price will rise or fall. So, we need to take the "prices" of the item we're trying to predict. 
#Let's stick with saying we're trying to predict the price of Litecoin. So we need to grab the future price of Litecoin, then determine if it's higher or lower to the current price. We need to do this at every step.

def classify(current,future):
	if float(future)> float(current):
		return 1
	else:
		return 0



main_df = pd.DataFrame()

ratios = ["BTC-USD","LTC-USD","ETH-USD","BCH-USD"]
for ratio in ratios:
	dataset = f"crypto_data/{ratio}.csv"
	df = pd.read_csv(dataset, names=["time","low","high","open","close","volume"])
	#print(df.head)
	df.rename(columns={"close":f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace=True)
	df.set_index("time",inplace=True)
	df=df[[f"{ratio}_close",f"{ratio}_volume"]]

	if len(main_df)==0:
		main_df=df
	else:
		main_df=main_df.join(df)

#A .shift will just shift the columns for us, a negative shift will shift them "up." So shifting up 3 will give us the price 3 minutes in the future, and we're just assigning this to a new column.
#each row has the price at the end of a minute, so a shift by 3 will give us the price 3 minute "in the future"
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

#Next, we need to create a target. To do this, we need to know which price we're trying to predict. We also need to know how far out we want to predict. 
#We'll go with Litecoin for now. Knowing how far out we want to predict probably also depends how long our sequences are. If our sequence length is 3 (so...3 minutes), 
#we probably can't easily predict out 10 minutes. If our sequence length is 300, 10 might not be as hard. I'd like to go with a sequence length of 60, and a future prediction out of 3.


#The map() is used to map a function. The first parameter here is the function we want to map (classify), then the next ones are the parameters to that function. In this case, the current close price, and then the future price.
#The map part is what allows us to do this row-by-row for these columns, but also do it quite fast. The list part converts the end result to a list, which we can just set as a column.

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"],  main_df["future"] ))

print( main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))
