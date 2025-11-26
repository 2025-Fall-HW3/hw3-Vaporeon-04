import yfinance as yf
import numpy as np
import pandas as pd
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2012-01-01"
end = "2024-04-01"

# download
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw['Adj Close']
df_returns = df.pct_change().fillna(0)

class MyStrategy:
    def __init__(self,
                 exclude="SPY",
                 lookback_mom=126,
                 vol_window=63,
                 top_k=6,   #4
                 target_vol=0.155):     # 🔥 明顯降低目標波動率
        self.exclude = exclude
        self.lookback_mom = lookback_mom
        self.vol_window = vol_window
        self.top_k = top_k
        self.target_vol = target_vol

        # 超低風險配置（熊市模式）
        self.bear_weights = {
            "XLP": 0.98,    #0.5-0.6
            "XLU": 0.01,    #0.3-0.2
            "XLV": 0.01,    #0.2
        }

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns).fillna(0.0)

        spy_ma200 = df["SPY"].rolling(200).mean()

        for i in range(max(self.lookback_mom, 200), len(df)):
            date = df.index[i]
            is_bull = df["SPY"].iloc[i] > spy_ma200.iloc[i]

            if not is_bull:
                # ===== 熊市模式：超低波防禦組合 =====
                w = pd.Series(0.0, index=assets)
                for sec, wt in self.bear_weights.items():
                    if sec in assets:
                        w[sec] = wt
                self.portfolio_weights.loc[date, assets] = w.values
                continue

            # ===== 多頭模式：動能 + 低波排序 =====
            hist = df_returns[assets].iloc[i - self.lookback_mom : i]
            vol_hist = df_returns[assets].iloc[i - self.vol_window : i]

            momentum = (1 + hist).prod() - 1
            vol = vol_hist.std().replace(0, np.nan)

            # 🔴 原始分數：score = (momentum / vol)
            # 🟢 新分數：只使用純動能作為排序依據
            score = momentum.replace([np.inf, -np.inf], np.nan).fillna(-9999) 

            # top_assets 仍選 top_k
            top_assets = score.sort_values(ascending=False).iloc[:self.top_k].index
            w = pd.Series(0.0, index=assets)
            w[top_assets] = 1.0 / len(top_assets)

            # ===== Minimum Variance Overlay（最重要）=====
            Sigma = vol_hist.cov().values + np.eye(len(assets)) * 1e-8
            w_vec = w.values
            port_vol = np.sqrt((w_vec @ Sigma @ w_vec) * 252)

            scale = self.target_vol / port_vol if port_vol > 0 else 1.0
            w = w * scale

            # 不允許 sector 過大
            w = w.clip(lower=0, upper=0.35)

            if w.sum() > 0:
                w = w / w.sum()

            self.portfolio_weights.loc[date, assets] = w.values

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = self.portfolio_returns[assets].mul(self.portfolio_weights[assets]).sum(axis=1)

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markowitz 2 - My Portfolio")
    parser.add_argument("--score", action="append", help="Score for assignment")
    args = parser.parse_args()

    judge_mode = False
    # If running in the grader environment, grader will call with --score flags
    # For local debugging, you can run the module to print metrics
    strat = MyStrategy()
    strat.calculate_portfolio_returns()
    w, perf = strat.get_results()

    # If grader is running, AssignmentJudge will inspect the output (protected file)
    try:
        from grader import AssignmentJudge
        judge = AssignmentJudge()
        judge.run_grading(args)
    except Exception:
        # Local quick report
        returns = perf["Portfolio"].dropna()
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        print(f"MyStrategy Annual Return: {ann_ret:.4f}, Vol: {ann_vol:.4f}, Sharpe: {sharpe:.4f}")
        # compare to SPY (2012-2024)
        spy = perf["SPY"].dropna()
        spy_ann_ret = spy.mean() * 252
        spy_ann_vol = spy.std() * np.sqrt(252)
        spy_sharpe = spy_ann_ret / spy_ann_vol if spy_ann_vol > 0 else np.nan
        print(f"SPY Annual Return: {spy_ann_ret:.4f}, Vol: {spy_ann_vol:.4f}, Sharpe: {spy_sharpe:.4f}")
