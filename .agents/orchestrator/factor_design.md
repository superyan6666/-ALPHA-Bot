# Factor Design Specification

The following 8 novel macroeconomic and valuation factors are designed to capture alpha on the CSI 300 Index using the merged dataset.

## 1. Equity Risk Premium (`F_ERP`)
- **Formula**:
  $$F\_ERP = \frac{100.0}{pe} - yield\_10y$$
- **Description**: Measures stock earnings yield relative to bond yields. A high ERP indicates stock market under-valuation compared to fixed income.
- **Expected Sign**: Positive (higher ERP should predict higher future stock returns).

## 2. VIX Term Structure (`F_VIX_TS`)
- **Formula**:
  $$F\_VIX\_TS = \frac{vix}{vix3m} - 1.0$$
- **Description**: Measures the ratio of short-term implied volatility (VIX) to medium-term implied volatility (VIX3M). During market panics, VIX spikes faster than VIX3M, creating buying opportunities when it reverses.
- **Expected Sign**: Positive (reversal signal).

## 3. PE Valuation Z-score (`F_PE_Zscore_252`)
- **Formula**:
  $$F\_PE\_Zscore\_252 = \frac{pe - MA(pe, 252)}{STD(pe, 252) + 1e-8}$$
- **Description**: Standardizes the current index PE ratio over a 1-year (252 trading days) rolling window to capture valuation extremes.
- **Expected Sign**: Negative (lower PE relative to historical average predicts higher future returns).

## 4. Bond Yield Momentum (`F_Yield_Momentum_20`)
- **Formula**:
  $$F\_Yield\_Momentum\_20 = yield\_10y_t - yield\_10y_{t-20}$$
- **Description**: Captures the rolling 20-day change in the China 10-Year Government Bond yield, serving as a proxy for rate shifts and liquidity trends.
- **Expected Sign**: Negative (rising rates typically pressure stock valuations).

## 5. PE-Yield Product Ratio (`F_PE_Yield_Ratio`)
- **Formula**:
  $$F\_PE\_Yield\_Ratio = pe \times yield\_10y$$
- **Description**: Combines stock and bond pricing into a unified valuation ratio (acting as a proxy for joint asset pricing pressures).
- **Expected Sign**: Negative (higher values indicate both stocks and bonds are expensive/yields are low/high PE).

## 6. VIX Momentum (`F_VIX_Momentum_10`)
- **Formula**:
  $$F\_VIX\_Momentum\_10 = vix_t - vix_{t-10}$$
- **Description**: Captures the rate of change in market fear over a 10-day window.
- **Expected Sign**: Positive (fear spikes followed by mean reversion).

## 7. VIX Volatility (`F_VIX_Volatility_20`)
- **Formula**:
  $$F\_VIX\_Volatility\_20 = STD(vix, 20)$$
- **Description**: Measures the historical volatility of the fear index over a 20-day rolling window, indicating volatility regime shifts.
- **Expected Sign**: Positive.

## 8. PE Moving Average Gap (`F_PE_Gap_120`)
- **Formula**:
  $$F\_PE\_Gap\_120 = \frac{pe}{MA(pe, 120)} - 1.0$$
- **Description**: Measures the percentage deviation of the PE ratio from its 120-day moving average.
- **Expected Sign**: Negative.
