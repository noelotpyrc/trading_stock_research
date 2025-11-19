# Context

We have minute bar and second bar OHLCV data for dozens of stocks under the folder /Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv, and all of them were picked around earnings report release events.

We want to develop some trading strategies based on these data, and we need to engineer some features for these data.

# First step

We want to use the earning report release time (acceptance_datetime_utc) as reference and segment the data into different time windows, and then calculate some features for each time window.

## Preprocessing step

Identify the trading hours of each day, and add a column to indicate which trading hour the data belongs to (before, after, or regular).

## Time windows

1. All the trading hours of N days before the earning report release event
2. 0-30 mins before the earning report release event, and 0-30 mins after the earning report release event
3. 30-240 mins after the earning report release event
4. 0-30 mins before and after the opening of the next regular hours of the earning report release event
5. All the trading hours of N days after the earning report release event

Note:
1. Earning report release could happen at any time of the day: 
    - if it happens in before hours or regular hours, N days before the earning report release event end at the previous day's after hours ending, and N days after the earning report release event start at the next day's before hours starting
    - if it happens in after hours, N days before the earning report release event end at the same trading day's regular hours ending, and N days after the earning report release event start at the next day's regular hours starting