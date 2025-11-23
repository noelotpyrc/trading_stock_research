# Earnings & Opening Strategies – Description + Pseudo-Code  
(1-second OHLCV data, low-latency execution)

### 1. Earnings Immediate Reaction Breakout (EIR)

**Description**  
As soon as the earnings report hits (typically 16:31–16:35 post-close or 07:30–08:30 pre-open), the first 30–90 seconds of trading are dominated by algos slamming the price in one direction. The high and low of that very first minute almost always define the short-term trend for the next 5–30 minutes. You simply wait for the price to break that tiny initial range on elevated volume and ride the momentum.

**Pseudo-code**
```
# PARAMETERS
OR_WINDOW_SEC        = 60          # opening range window (30–90s typical)
VOL_MA_WINDOW_SEC    = 20          # rolling volume lookback
VOL_SPIKE_MULT       = 3.0         # 3x average volume
MAX_TRADE_WINDOW_MIN = 5          # trade up to 5 min after earnings

USE_CLOSE_CONFIRM    = true        # require close beyond OR levels

# STATE
earnings_time
OR_high, OR_low
or_defined          = false

long_taken          = false
short_taken         = false

rolling_vol_buffer  = []           # volumes for last VOL_MA_WINDOW_SEC seconds


# 1) At earnings release
on_earnings_event(time_event):
    earnings_time = time_event
    or_defined    = false
    long_taken    = false
    short_taken   = false
    clear(rolling_vol_buffer)


# 2) Build opening range (OR) from first OR_WINDOW_SEC seconds
on_new_1s_bar(t, high, low, close, volume):

    # Stop trading after max window
    if t > earnings_time + MAX_TRADE_WINDOW_MIN * 60:
        return

    # Update rolling volume buffer
    append(rolling_vol_buffer, volume, keep_last = VOL_MA_WINDOW_SEC)
    if length(rolling_vol_buffer) < VOL_MA_WINDOW_SEC:
        vol_ma = null
    else:
        vol_ma = mean(rolling_vol_buffer)

    # --- Step A: define opening range ---
    seconds_since_earnings = t - earnings_time

    if seconds_since_earnings <= OR_WINDOW_SEC:
        # still forming OR
        if not or_defined:
            if seconds_since_earnings == 1:
                OR_high = high
                OR_low  = low
            else:
                OR_high = max(OR_high, high)
                OR_low  = min(OR_low, low)
        return

    # After OR_WINDOW_SEC, OR is locked in
    or_defined = true


    # --- Step B: trade logic after OR is defined ---

    if vol_ma is null:
        return     # need enough data to define volume average

    # Volume spike condition
    vol_spike = (volume > VOL_SPIKE_MULT * vol_ma)

    # Breakout checks
    if USE_CLOSE_CONFIRM:
        long_break  = (close > OR_high)
        short_break = (close < OR_low)
    else:
        long_break  = (high > OR_high)
        short_break = (low  < OR_low)

    # LONG breakout (only once)
    if not long_taken and long_break and vol_spike:
        enter_long_at_market()
        stop_loss   = OR_low - small_buffer
        target_dist = 3 * (OR_high - OR_low)
        take_profit = entry_price + target_dist
        long_taken  = true

    # SHORT breakout (only once)
    if not short_taken and short_break and vol_spike:
        enter_short_at_market()
        stop_loss   = OR_high + small_buffer
        target_dist = 3 * (OR_high - OR_low)
        take_profit = entry_price - target_dist
        short_taken = true

    # Exit logic for any open position:
    manage_open_position(t, last_price = close)

```

### 1.1 Simple “Impulse Bar” Breakout After Earnings

**Description**  
Use just one big impulse bar right after the report as your reference, then trade a break of that impulse range.
- After the earnings time, watch for the first 1-second bar (or group of a few seconds) with volume spike and large range.
- Mark that bar’s high/low as the “impulse box.”
- Trade the first break of that box in the direction of the break, with no retest needed.

**Pseudo-code**
```
# PARAMETERS
IMPULSE_VOL_MULT    = 3.0       # vs 20-sec vol MA
IMPULSE_RANGE_MIN   = X ticks   # minimum bar size (custom)
VOL_MA_SEC          = 20
MAX_TRADE_MIN       = 20

earnings_time
rolling_vol   = []
impulse_found = false
IB_high, IB_low
trade_taken   = false

on_earnings_event(t_event):
    earnings_time  = t_event
    clear(rolling_vol)
    impulse_found = false
    trade_taken   = false

on_new_1s_bar(t, high, low, close, volume):

    if t > earnings_time + MAX_TRADE_MIN * 60:
        return

    append(rolling_vol, volume, keep_last = VOL_MA_SEC)
    if len(rolling_vol) < VOL_MA_SEC:
        return

    vol_ma = mean(rolling_vol)
    vol_spike = (volume > IMPULSE_VOL_MULT * vol_ma)
    bar_range = high - low

    # 1) Detect impulse bar
    if not impulse_found and vol_spike and bar_range >= IMPULSE_RANGE_MIN:
        IB_high = high
        IB_low  = low
        impulse_found = true
        return

    if not impulse_found:
        return

    if trade_taken:
        manage_position(close)
        return

    # 2) Trade first clean break of impulse range
    if close > IB_high:
        enter LONG at market
        stop   = IB_low - buffer
        target = close + 2–3 * (IB_high - IB_low)
        trade_taken = true

    else if close < IB_low:
        enter SHORT at market
        stop   = IB_high + buffer
        target = close - 2–3 * (IB_high - IB_low)
        trade_taken = true

    manage_position(close)

```

### 2. Cumulative Volume Delta (CVD) Momentum Continuation

**Description**  
After the initial knee-jerk reaction, check if aggressive buyers or sellers are still dominating. If buying pressure (positive delta) keeps pouring in and price stays above the developing VWAP, the move has legs. Second-level data lets you see this divergence or confirmation instantly.

**Pseudo-code for delta calculation**
```
# On every new 1-second bar:
# inputs: high, low, close, volume

if high == low:
    delta_this_bar = 0          # no directional info
else:
    midpoint   = (high + low) / 2
    half_range = (high - low) / 2

    distance_from_mid = (close - midpoint) / half_range   # -∞..+∞
    distance_from_mid = clamp(distance_from_mid, -1.0, 1.0)

    delta_this_bar = volume * distance_from_mid
```

**Pseudo-code**
```
# PARAMETERS
WAIT_SEC_AFTER_EVENT   = 90         # wait after report/open
MIN_SAMPLES_FOR_Z      = 60         # min 60 seconds of CVD data
Z_LONG_THRESHOLD       = 1.2        # example
Z_SHORT_THRESHOLD      = -1.2
DELTA_SLOPE_LOOKBACK   = 20         # 20s for "still rising/falling"
EMA_FAST_SEC           = 8          # entry filter
EMA_EXIT_SEC           = 5          # exit filter
DELTA_EXIT_SEC         = 10         # for exit
TRADE_TIME_STOP_MIN    = 15         # 15–20 min window

# STATE
event_time      # earnings time or 9:30 open
cvd             # cumulative delta since event
cvd_history[]   # list of cvd snapshots each second
delta_buffer[]  # last ~20–30s of per-second delta
position_side   # 'FLAT', 'LONG', 'SHORT'
entry_time, entry_price, etc.

# ------------------------------
on_event(time_event):
    event_time    = time_event
    cvd           = 0
    clear(cvd_history)
    clear(delta_buffer)
    position_side = 'FLAT'

# ------------------------------
on_new_second_bar(t, price, delta):

    # 1) Update CVD & buffers
    cvd += delta
    append(cvd_history, cvd)
    append(delta_buffer, delta, keep_last = max(DELTA_SLOPE_LOOKBACK, DELTA_EXIT_SEC))

    update_VWAP_since_event()
    update_EMA(price, EMA_FAST_SEC)
    update_EMA(price, EMA_EXIT_SEC)

    # 2) Too early to trade? just manage exits
    if t < event_time + WAIT_SEC_AFTER_EVENT:
        manage_exits(t, price)
        return

    # 3) Need enough CVD samples to compute z-score
    if length(cvd_history) < MIN_SAMPLES_FOR_Z:
        manage_exits(t, price)
        return

    mean_cvd = mean(cvd_history)
    std_cvd  = std(cvd_history)
    if std_cvd is very small:
        manage_exits(t, price)
        return

    z_cvd = (cvd - mean_cvd) / std_cvd
    delta_last_20s = sum(last DELTA_SLOPE_LOOKBACK deltas from delta_buffer)

    if position_side == 'FLAT':
        try_open_with_zscore(t, price, z_cvd, delta_last_20s)
    else:
        manage_exits(t, price, z_cvd, delta_last_20s)

# ------------------------------
try_open_with_zscore(t, price, z_cvd, delta_last_20s):

    # LONG: strong positive CVD regime + recent buying + price filters
    if z_cvd >= Z_LONG_THRESHOLD
       and delta_last_20s > 0
       and price > VWAP
       and price > EMA_fast:
           enter_LONG(t, price)

    # SHORT: symmetric
    if z_cvd <= Z_SHORT_THRESHOLD
       and delta_last_20s < 0
       and price < VWAP
       and price < EMA_fast:
           enter_SHORT(t, price)

# ------------------------------
manage_exits(t, price, z_cvd = optional, delta_last_10s = optional):

    if position_side == 'FLAT':
        return

    # 1) delta-based exit
    delta_last_10s = sum(last DELTA_EXIT_SEC deltas from delta_buffer)
    if position_side == 'LONG' and delta_last_10s < 0:
        exit_all("DELTA_FLIP")
    if position_side == 'SHORT' and delta_last_10s > 0:
        exit_all("DELTA_FLIP")

    # 2) EMA exit
    if position_side == 'LONG' and price < EMA_exit:
        exit_all("EMA_EXIT")
    if position_side == 'SHORT' and price > EMA_exit:
        exit_all("EMA_EXIT")

    # 3) Time stop
    if t - entry_time >= TRADE_TIME_STOP_MIN:
        exit_all("TIME_STOP")
```

### 3. Post-Earnings Overreaction Fade (Mean Reversion)

**Description**  
- After the earnings event time, watch the first 5 minutes.
- If price makes a big move (> X%) in that 5-min window, treat it as a potential overreaction.
- After that, wait for:
    - Volume to collapse vs peak 1-min volume from the first 5 minutes
    - Price to be far from 5-min VWAP (in standard deviation units)
    - Short-term RSI(3) to be extreme
- Fade back toward VWAP / value area.

**Pseudo-code**
```
# PARAMETERS
MOVE_MIN_PCT      = 0.05        # 5%
VOL_DROP_RATIO    = 0.50        # < 50% of peak 1m vol
DIST_Z_THRESHOLD  = 2.5         # 2.5 std from VWAP
RSI_OB            = 85
RSI_OS            = 15
FIRST_5M_SEC      = 5 * 60
LOOKBACK_STD_SEC  = 5 * 60
MAX_TRADE_MIN     = 30

# STATE
earnings_time
first_price
first_5m_high, first_5m_low
peak_1m_vol
first_window_done = false
signal_taken      = false

on_earnings_event(t_event, price_event):
    earnings_time      = t_event
    first_price        = price_event
    first_5m_high      = price_event
    first_5m_low       = price_event
    peak_1m_vol        = 0
    first_window_done  = false
    signal_taken       = false

on_new_1s_bar(t, price, volume):

    if t > earnings_time + MAX_TRADE_MIN * 60:
        return

    seconds_since = t - earnings_time

    update_rolling_1m_volume(volume)          # returns vol_1m_now
    vol_1m_now = rolling_1m_volume

    update_VWAP_since(earnings_time)          # for later VWAP windows
    vwap_5m    = VWAP_over_window(earnings_time, earnings_time + FIRST_5M_SEC)
    std_5m     = std_price_over_last(LOOKBACK_STD_SEC)
    rsi_3      = RSI_1s(period=3)

    # ---- Phase 1: first 5 minutes ----
    if not first_window_done:
        first_5m_high  = max(first_5m_high, price)
        first_5m_low   = min(first_5m_low,  price)
        peak_1m_vol    = max(peak_1m_vol,   vol_1m_now)

        if seconds_since >= FIRST_5M_SEC:
            first_window_done = true
        return

    # ---- Phase 2: overreaction fade ----
    if signal_taken or peak_1m_vol <= 0 or std_5m <= 0:
        return

    move_up_pct   = (first_5m_high - first_price) / first_price
    move_down_pct = (first_5m_low  - first_price) / first_price
    big_up        = (move_up_pct   >= MOVE_MIN_PCT)
    big_down      = (move_down_pct <= -MOVE_MIN_PCT)
    if not big_up and not big_down:
        return

    vol_collapse = (vol_1m_now < VOL_DROP_RATIO * peak_1m_vol)
    if not vol_collapse:
        return

    dist      = price - vwap_5m
    dist_z    = abs(dist) / std_5m
    if dist_z <= DIST_Z_THRESHOLD:
        return

    # SHORT: big up + stretched above VWAP + overbought
    if big_up and price > vwap_5m and rsi_3 > RSI_OB:
        enter SHORT at market
        stop_loss   = first_5m_high + buffer
        target_vwap = VWAP_over_window(earnings_time, earnings_time + 15 * 60)
        take_profit = target_vwap   # or opposite side of initial gap
        signal_taken = true
        return

    # LONG: big down + stretched below VWAP + oversold
    if big_down and price < vwap_5m and rsi_3 < RSI_OS:
        enter LONG at market
        stop_loss   = first_5m_low - buffer
        target_vwap = VWAP_over_window(earnings_time, earnings_time + 15 * 60)
        take_profit = target_vwap
        signal_taken = true
        return

```

### 4. Regular Session Opening Range Breakout — Enhanced for Post-Earnings Days

**Description**  
Two phases:

1. Earnings day (phase 1) – after the earnings event, you analyze the reaction:
    - Direction & strength of price move
    - Direction & strength of CVD
    - How price behaves vs VWAP
    → From this you mark the stock as BULLISH_BIAS, BEARISH_BIAS, or NO_TRADE for the next day’s open.

2. Next trading day open (phase 2) – you run a trend-following intraday strategy at 9:30 that:
    - Only trades in the direction of the stored bias
    - Requires early order flow (CVD) and price action to confirm that bias

**Pseudo-code**
```
Phase 1: Earnings day: compute bias for next open
# PARAMS
POST_EVENT_DELAY       = 60s
MIN_ANALYSIS_MIN       = 30
PRICE_REACT_MIN_PCT    = 3%
CVD_Z_BULL             = 1.0
CVD_Z_BEAR             = -1.0
VWAP_UPTIME_MIN        = 0.60

# STATE
event_time
event_price
cvd_event = 0
cvd_series   = []
price_series = []
vwap_series  = []
bias_next_open = 'NONE'

on_earnings_event(t_event, price_event):
    event_time  = t_event
    event_price = price_event
    cvd_event   = 0
    clear(cvd_series, price_series, vwap_series)
    bias_next_open = 'NONE'

on_new_second_bar_earnings(t, price, delta):
    if t < event_time + POST_EVENT_DELAY:
        return

    cvd_event += delta
    append(cvd_series, cvd_event)
    append(price_series, price)
    append(vwap_series, VWAP_since_event())

on_earnings_close(t_close):
    if minutes_between(event_time + POST_EVENT_DELAY, t_close) < MIN_ANALYSIS_MIN:
        bias_next_open = 'NONE'
        return

    price_close = last(price_series)
    price_react = (price_close - event_price) / event_price

    mean_cvd = mean(cvd_series)
    std_cvd  = std(cvd_series)
    if std_cvd <= 0:
        bias_next_open = 'NONE'
        return

    z_cvd = (last(cvd_series) - mean_cvd) / std_cvd

    above = count(i: price_series[i] > vwap_series[i])
    vwap_uptime = above / len(price_series)

    if price_react >= PRICE_REACT_MIN_PCT
       and z_cvd >= CVD_Z_BULL
       and vwap_uptime >= VWAP_UPTIME_MIN:
           bias_next_open = 'BULLISH'
    else if price_react <= -PRICE_REACT_MIN_PCT
       and z_cvd <= CVD_Z_BEAR
       and (1 - vwap_uptime) >= VWAP_UPTIME_MIN:
           bias_next_open = 'BEARISH'
    else:
           bias_next_open = 'NONE'

Phase 2 Next day open: trade with bias

# PARAMS
WAIT_AFTER_OPEN_SEC  = 60–120
MIN_Z_SAMPLES        = 60
Z_CONFIRM            = 1.0
DELTA_SLOPE_SEC      = 20
DELTA_EXIT_SEC       = 10
EMA_FAST_SEC         = 8
EMA_EXIT_SEC         = 5
TIME_STOP_MIN        = 15–20

# STATE
open_time
cvd_today = 0
cvd_today_series   = []
delta_today_buffer = []
position_side = 'FLAT'
entry_time   = null
entry_price  = null

on_next_day_open(t_open, open_price):
    open_time = t_open
    cvd_today = 0
    clear(cvd_today_series, delta_today_buffer)
    position_side = 'FLAT'

on_new_second_bar_next(t, price, delta):

    # update flows
    cvd_today += delta
    append(cvd_today_series, cvd_today)
    append(delta_today_buffer, delta, keep_last = max(DELTA_SLOPE_SEC, DELTA_EXIT_SEC))
    update_VWAP_today()
    update_EMA(price, EMA_FAST_SEC)
    update_EMA(price, EMA_EXIT_SEC)

    if bias_next_open == 'NONE':
        manage_exits(t, price)
        return

    if t < open_time + WAIT_AFTER_OPEN_SEC:
        manage_exits(t, price)
        return

    if len(cvd_today_series) < MIN_Z_SAMPLES:
        manage_exits(t, price)
        return

    mean_cvd = mean(cvd_today_series)
    std_cvd  = std(cvd_today_series)
    if std_cvd <= 0:
        manage_exits(t, price)
        return

    z_cvd   = (cvd_today - mean_cvd) / std_cvd
    d20s    = sum(last DELTA_SLOPE_SEC deltas)
    d10s    = sum(last DELTA_EXIT_SEC deltas)

    if position_side == 'FLAT':
        # entries
        if bias_next_open == 'BULLISH'
           and z_cvd >= Z_CONFIRM
           and d20s > 0
           and price > VWAP_today
           and price > EMA_fast:
               enter_LONG(t, price)

        else if bias_next_open == 'BEARISH'
           and z_cvd <= -Z_CONFIRM
           and d20s < 0
           and price < VWAP_today
           and price < EMA_fast:
               enter_SHORT(t, price)

    else:
        # exits
        if position_side == 'LONG' and d10s < 0:
            exit_all("DELTA_FLIP")
        else if position_side == 'SHORT' and d10s > 0:
            exit_all("DELTA_FLIP")
        else if position_side == 'LONG' and price < EMA_exit:
            exit_all("EMA_EXIT")
        else if position_side == 'SHORT' and price > EMA_exit:
            exit_all("EMA_EXIT")
        else if minutes_between(entry_time, t) >= TIME_STOP_MIN:
            exit_all("TIME_STOP")

```
