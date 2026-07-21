
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class MonteCarloSimulator:
    
    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        self.n_sims = n_simulations
        self.rng = np.random.RandomState(seed)

    # ═══════════════════════════════════════════════════════════
    #  SIMULATION METHODS
    # ═══════════════════════════════════════════════════════════

    def run_block_bootstrap(self, daily_returns: np.ndarray,
                            block_size: int = 20,
                            initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) < block_size:
            return self._empty_result()

        n = len(daily_returns)
        n_blocks = max(1, n // block_size)

        # Generate all block-start indices for every sim at once (2D batch)
        all_starts = self.rng.randint(0, n - block_size, size=(self.n_sims, n_blocks))

        # Build all simulated return series as a 2D array (n_sims × n)
        all_sim_returns = np.array([
            # Sample random block start indices
            np.concatenate([daily_returns[s:s + block_size] for s in starts])[:n]
            for starts in all_starts
        ])  # Trim to original length

        # Vectorised equity curves via cumprod (n_sims × n+1)
        all_equity = initial_capital * np.column_stack([
            np.ones(self.n_sims),
            np.cumprod(1 + all_sim_returns, axis=1)
        ])

        # Vectorised final returns, drawdowns, sharpes
        all_final_returns = (all_equity[:, -1] - initial_capital) / initial_capital
        all_max_dd = np.array([self._max_drawdown(eq) for eq in all_equity])
        all_sharpe = np.array([self._sharpe(sr) for sr in all_sim_returns])

        return {
            "equity_curves": list(all_equity),
            "returns": all_final_returns,
            "max_drawdowns": all_max_dd,
            "sharpe_ratios": all_sharpe,
            "n_sims": self.n_sims,
            "method": "block_bootstrap",
        }

    def run_regime_aware(self, daily_returns: np.ndarray,
                         volatility: np.ndarray = None,
                         initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) < 40:
            return self._empty_result()

        n = len(daily_returns)

        # Compute rolling volatility if not provided
        if volatility is None:
            volatility = pd.Series(daily_returns).rolling(20, min_periods=5).std().fillna(
                pd.Series(daily_returns).std()
            ).values

        # Classify into regimes: high vol vs low vol
        vol_median = np.median(volatility)
        high_vol_mask = volatility > vol_median
        low_vol_mask = ~high_vol_mask

        high_vol_returns = daily_returns[high_vol_mask]
        low_vol_returns = daily_returns[low_vol_mask]

        if len(high_vol_returns) == 0 or len(low_vol_returns) == 0:
            return self.run_block_bootstrap(daily_returns, initial_capital=initial_capital)

        n_high = int(high_vol_mask.sum())
        n_low = n - n_high

        # Pre-sample all regime-aware returns for every sim at once (2D batch)
        high_samples = self.rng.choice(high_vol_returns, size=(self.n_sims, n_high), replace=True)
        low_samples = self.rng.choice(low_vol_returns, size=(self.n_sims, n_low), replace=True)

        # Assemble full return arrays by scattering regime samples into position
        all_sim_returns = np.empty((self.n_sims, n))
        all_sim_returns[:, high_vol_mask] = high_samples
        all_sim_returns[:, low_vol_mask] = low_samples

        # Add small noise for variation
        all_sim_returns += self.rng.normal(0, 0.0005, size=(self.n_sims, n))

        # Vectorised equity curves via cumprod (n_sims × n+1)
        all_equity = initial_capital * np.column_stack([
            np.ones(self.n_sims),
            np.cumprod(1 + all_sim_returns, axis=1)
        ])

        # Vectorised final returns, drawdowns, sharpes
        all_final_returns = (all_equity[:, -1] - initial_capital) / initial_capital
        all_max_dd = np.array([self._max_drawdown(eq) for eq in all_equity])
        all_sharpe = np.array([self._sharpe(sr) for sr in all_sim_returns])

        return {
            "equity_curves": list(all_equity),
            "returns": all_final_returns,
            "max_drawdowns": all_max_dd,
            "sharpe_ratios": all_sharpe,
            "n_sims": self.n_sims,
            "method": "regime_aware",
        }

    def run_trade_resample(self, trade_returns: np.ndarray,
                           n_trades_per_sim: int = None,
                           initial_capital: float = 100_000) -> dict:
        
        if len(trade_returns) == 0:
            return self._empty_result()

        n_trades = n_trades_per_sim or len(trade_returns)

        # Batch-sample all trade returns for every sim at once (2D)
        all_sampled = self.rng.choice(trade_returns, size=(self.n_sims, n_trades), replace=True)

        # Vectorised equity curves via cumprod (n_sims × n_trades+1)
        all_equity = initial_capital * np.column_stack([
            np.ones(self.n_sims),
            np.cumprod(1 + all_sampled, axis=1)
        ])

        # Vectorised final returns, drawdowns, sharpes
        all_final_returns = (all_equity[:, -1] - initial_capital) / initial_capital
        all_max_dd = np.array([self._max_drawdown(eq) for eq in all_equity])
        all_sharpe = np.array([self._sharpe(sr) for sr in all_sampled])

        return {
            "equity_curves": list(all_equity),
            "returns": all_final_returns,
            "max_drawdowns": all_max_dd,
            "sharpe_ratios": all_sharpe,
            "n_sims": self.n_sims,
            "method": "trade_resample",
        }

    def run_return_perturbation(self, daily_returns: np.ndarray,
                                noise_std: float = 0.001,
                                initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) == 0:
            return self._empty_result()

        n = len(daily_returns)

        # Generate all noise for every sim at once (2D batch)
        all_noise = self.rng.normal(0, noise_std, size=(self.n_sims, n))
        all_perturbed = daily_returns[np.newaxis, :] + all_noise

        # Vectorised equity curves via cumprod (n_sims × n+1)
        all_equity = initial_capital * np.column_stack([
            np.ones(self.n_sims),
            np.cumprod(1 + all_perturbed, axis=1)
        ])

        # Vectorised final returns, drawdowns, sharpes
        all_final_returns = (all_equity[:, -1] - initial_capital) / initial_capital
        all_max_dd = np.array([self._max_drawdown(eq) for eq in all_equity])
        all_sharpe = np.array([self._sharpe(sr) for sr in all_perturbed])

        return {
            "equity_curves": list(all_equity),
            "returns": all_final_returns,
            "max_drawdowns": all_max_dd,
            "sharpe_ratios": all_sharpe,
            "n_sims": self.n_sims,
            "method": "return_perturbation",
        }

    def run_cost_variation(self, daily_returns: np.ndarray,
                           base_cost_per_trade: float = 0.0006,
                           cost_std: float = 0.0003,
                           avg_trades_per_day: float = 0.1,
                           initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) == 0:
            return self._empty_result()

        all_equity = []
        all_final_returns = []
        all_max_dd = []

        for _ in range(self.n_sims):
            # Pre-generate all random decisions at once
            trade_mask = self.rng.random(size=len(daily_returns)) < avg_trades_per_day
            costs = np.where(
                trade_mask,
                np.maximum(0, self.rng.normal(base_cost_per_trade, cost_std,
                                              size=len(daily_returns))),
                0.0
            )
            adjusted_returns = daily_returns - costs

            # Vectorised equity curve via cumprod
            equity = initial_capital * np.concatenate([
                [1.0], np.cumprod(1 + adjusted_returns)
            ])

            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))

        return {
            "equity_curves": all_equity,
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array([]),
            "n_sims": self.n_sims,
            "method": "cost_variation",
        }

    def run_stress_test(self, daily_returns: np.ndarray,
                        crash_magnitude: float = -0.05,
                        crash_probability: float = 0.01,
                        spread_explosion_mult: float = 5.0,
                        initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) == 0:
            return self._empty_result()

        n = len(daily_returns)
        all_equity = []
        all_final_returns = []
        all_max_dd = []

        for _ in range(self.n_sims):
            adjusted = daily_returns.copy()

            # Pre-generate all crash events at once
            crash_mask = self.rng.random(size=n) < crash_probability
            # Random flash crash
            adjusted[crash_mask] += crash_magnitude

            # Pre-generate all spread explosion events at once
            spread_mask = self.rng.random(size=n) < crash_probability * 2
            # Random spread explosion
            adjusted[spread_mask] -= 0.001 * spread_explosion_mult

            # Vectorised equity curve via cumprod
            equity = initial_capital * np.concatenate([
                [1.0], np.cumprod(1 + adjusted)
            ])

            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))

        return {
            "equity_curves": all_equity,
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array([]),
            "n_sims": self.n_sims,
            "method": "stress_test",
        }

    # ═══════════════════════════════════════════════════════════
    #  REPORTING
    # ═══════════════════════════════════════════════════════════

    def generate_report(self, result: dict) -> dict:
        
        returns = result["returns"]
        drawdowns = result["max_drawdowns"]
        sharpes = result.get("sharpe_ratios", np.array([]))

        report = {
            "method": result["method"],
            "n_simulations": result["n_sims"],
            # Returns
            "mean_return": np.mean(returns),
            "median_return": np.median(returns),
            "std_return": np.std(returns),
            "return_5th_pct": np.percentile(returns, 5),
            "return_25th_pct": np.percentile(returns, 25),
            "return_75th_pct": np.percentile(returns, 75),
            "return_95th_pct": np.percentile(returns, 95),
            # Risk
            "prob_positive": np.mean(returns > 0),
            "prob_ruin_10pct": np.mean(returns < -0.10),
            "prob_ruin_20pct": np.mean(returns < -0.20),
            "mean_max_dd": np.mean(drawdowns),
            "worst_max_dd": np.min(drawdowns),
            "dd_95th_pct": np.percentile(drawdowns, 5),
        }

        if len(sharpes) > 0:
            report["mean_sharpe"] = np.mean(sharpes)
            report["median_sharpe"] = np.median(sharpes)
            report["sharpe_5th_pct"] = np.percentile(sharpes, 5)
            report["sharpe_95th_pct"] = np.percentile(sharpes, 95)

        return report

    def plot_all(self, result: dict, save_dir: str = "results/monte_carlo"):
        
        os.makedirs(save_dir, exist_ok=True)

        self._plot_equity_fan(result, save_dir)
        self._plot_return_histogram(result, save_dir)
        self._plot_drawdown_histogram(result, save_dir)
        if len(result.get("sharpe_ratios", [])) > 0:
            self._plot_sharpe_histogram(result, save_dir)

        print(f"[MC] All plots saved to {save_dir}")

    # ═══════════════════════════════════════════════════════════
    #  PLOT FUNCTIONS
    # ═══════════════════════════════════════════════════════════

    def _plot_equity_fan(self, result: dict, save_dir: str):
        
        curves = result["equity_curves"]
        if isinstance(curves, list):
            max_len = min(len(c) for c in curves)
            curves_arr = np.array([c[:max_len] for c in curves])
        else:
            curves_arr = curves
            max_len = curves_arr.shape[1]

        x = np.arange(max_len)
        p5 = np.percentile(curves_arr, 5, axis=0)
        p25 = np.percentile(curves_arr, 25, axis=0)
        p50 = np.percentile(curves_arr, 50, axis=0)
        p75 = np.percentile(curves_arr, 75, axis=0)
        p95 = np.percentile(curves_arr, 95, axis=0)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.fill_between(x, p5, p95, alpha=0.15, color='steelblue', label='5-95th pct')
        ax.fill_between(x, p25, p75, alpha=0.3, color='steelblue', label='25-75th pct')
        ax.plot(x, p50, color='navy', lw=2, label='Median')

        # Find worst and best paths
        final_values = curves_arr[:, -1]
        worst_idx = np.argmin(final_values)
        best_idx = np.argmax(final_values)
        ax.plot(x, curves_arr[worst_idx], color='red', lw=0.8, alpha=0.5, label='Worst path')
        ax.plot(x, curves_arr[best_idx], color='green', lw=0.8, alpha=0.5, label='Best path')

        ax.axhline(curves_arr[0, 0], color='gray', ls=':', lw=1, alpha=0.5)
        ax.set_title(f"Monte Carlo Equity Fan ({result['n_sims']} sims — {result['method']})",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_equity_fan.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_return_histogram(self, result: dict, save_dir: str):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = result["returns"] * 100

        ax.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='navy', lw=0.5)
        ax.axvline(np.mean(returns), color='red', ls='--', lw=2, label=f'Mean: {np.mean(returns):.1f}%')
        ax.axvline(np.median(returns), color='orange', ls='--', lw=2, label=f'Median: {np.median(returns):.1f}%')
        ax.axvline(0, color='black', ls='-', lw=1)

        prob_pos = np.mean(result["returns"] > 0) * 100
        ax.set_title(f"MC Return Distribution ({prob_pos:.0f}% positive)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Total Return (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_return_hist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_drawdown_histogram(self, result: dict, save_dir: str):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        dd = result["max_drawdowns"] * 100

        ax.hist(dd, bins=50, color='#e74c3c', alpha=0.7, edgecolor='darkred', lw=0.5)
        ax.axvline(np.mean(dd), color='navy', ls='--', lw=2, label=f'Mean: {np.mean(dd):.1f}%')
        ax.axvline(np.percentile(dd, 5), color='black', ls=':', lw=2, label=f'5th pct: {np.percentile(dd, 5):.1f}%')

        ax.set_title("MC Max Drawdown Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_drawdown_hist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_sharpe_histogram(self, result: dict, save_dir: str):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sharpes = result["sharpe_ratios"]

        ax.hist(sharpes, bins=50, color='#2ecc71', alpha=0.7, edgecolor='darkgreen', lw=0.5)
        ax.axvline(np.mean(sharpes), color='red', ls='--', lw=2, label=f'Mean: {np.mean(sharpes):.2f}')
        ax.axvline(0, color='black', ls='-', lw=1)
        ax.axvline(np.percentile(sharpes, 5), color='gray', ls=':', lw=2,
                   label=f'5th pct: {np.percentile(sharpes, 5):.2f}')

        prob_pos = np.mean(sharpes > 0) * 100
        ax.set_title(f"MC Sharpe Distribution ({prob_pos:.0f}% positive)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_sharpe_hist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  STATIC HELPERS
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _max_drawdown(equity):
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        return dd.min()

    @staticmethod
    def _sharpe(returns, periods=252):
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(periods)

    @staticmethod
    def _empty_result():
        return {
            "equity_curves": [np.array([100_000])],
            "returns": np.array([0.0]),
            "max_drawdowns": np.array([0.0]),
            "sharpe_ratios": np.array([0.0]),
            "n_sims": 0,
            "method": "empty",
        }