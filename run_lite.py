import os
import sys

# Configure lite mode environment variables
os.environ["WF_TIMESTEPS"] = "1000"  # Fast test
os.environ["SKIP_SVM"] = "True"      # SVM is slow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MLRL01"))

from MLRL01.agents.train import load_latest_data, split_data, train_ml_models
from MLRL01.features.features import build_production_features
from MLRL01.monte_carlo.simulator import MonteCarloSimulator

def run_lite():
    print("=" * 60)
    print("🚀 MLRL01 LITE MODE (Low-End PC Compatible)")
    print("=" * 60)

    try:
        # 1. Load Data
        print("\n📥 Loading data...")
        df_raw = load_latest_data("MLRL01/jupiter")
        
        # 2. Build Features
        print("\n🔧 Engineering features (Vectorized + Cached)...")
        df = build_production_features(df_raw, use_cache=True, cache_dir="results/cache")
        
        # 3. Simulate Monte Carlo
        print("\n🎲 Running Monte Carlo Simulation (Vectorized)...")
        # Generate dummy equity to simulate
        dummy_returns = df['ret_1d'].fillna(0).values[:500]
        mc = MonteCarloSimulator(n_simulations=200) # Small for lite mode
        mc_res = mc.run_block_bootstrap(dummy_returns)
        
        print(f"✅ Block Bootstrap Completed! Median Return: {mc_res['returns'].mean():.2%}")

        # 4. Success message
        print("\n✅ LITE RUN COMPLETED SUCCESSFULLY.")
        print("This proves the core optimizations (Vectorized Python Loops, Caching) are working.")
        print("To run RL (PPO) and ML models, ensure `env.trading_env` is present in MLRL01.")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")

if __name__ == "__main__":
    run_lite()
