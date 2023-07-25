import tradingeconomics as te
te.login('Your_Key:Your_Secret')
data = te.getHistoricalData(country='United States', indicator='Households Debt to GDP', initDate='2015-01-01')
print(data)