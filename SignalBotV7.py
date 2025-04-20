import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from datetime import datetime, timedelta
import time
from collections import deque
import warnings
import statistics
from scipy import stats
import math
import json
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import pickle
import threading
import os
import sys
from dotenv import load_dotenv


########## see link below for more details on the new features and changes made to the bot ##########
####### https://claude.ai/chat/58c798e8-8ebd-44e2-b766-1707872000ca #######
###### Run again of claudeparams1 with actual adjusted signal weights below in signalbotV2_Claude_Params_3.py ######
####### Run 2 after decent data, updated SL and TP distances, quality requirements ######
##### https://claude.ai/chat/5a7d1ed8-457c-423f-9513-5358f8a72c37 ######
####### CHANGED TP/SL TO % FOR TESTING ####### AND SMALLER TP

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
wallet_address = os.environ.get("HYPERLIQUID_ADDRESS_LIVE_2")
private_key = os.environ.get("HYPERLIQUID_KEY_LIVE_2")

# Add a check to make sure the environment variables are loaded
if not wallet_address or not private_key:
    print("Error: Missing credentials. Check your .env file.")
    exit(1)

# Suppress warnings
warnings.filterwarnings("ignore")

# Advanced logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("ComboBotV6_retest.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SignalBot")


class SignalBot:
    """
    Adaptive trading bot that identifies and exploits liquidity levels with dynamic parameters
    based on market regimes and automated regime detection.
    """
    ##################################################################################################################################################
    ######################################################## Core Initialization and Setup ###########################################################
    ##################################################################################################################################################

    def __init__(
        self,
        symbol="ETH/USDC:USDC",
        timeframe="5m",
        risk_per_trade=0.01,
        leverage=1,
        wallet_address=None,
        private_key=None,
        debug_mode=True,
        exchange=None,
        base_sl_percent=1.0,
        base_tp_percent=1.0,
    ):
        """
        Enhanced initialization for the SignalBot with full functionality
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.debug_mode = debug_mode

        # Market data containers
        self.data = None
        self.orderbook_data = None
        self.orderbook_history = deque(maxlen=100)
        self.trades_history = deque(maxlen=200)
        self.l2_heatmap = None
        self.market_snapshot = None

        # Trading state
        self.active_trade = None
        self.last_liquidity_analysis = None
        self.market_regime = None
        self.volume_profile = None

        # Error control parameters
        self.safety_margin = 0.95
        self.position_check_interval = 5
        self.last_position_check = None
        self.error_recovery_mode = False

        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "drawdown": 0,
            "total_trades": 0,
            "total_profit_pct": 0,	
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
        }

        # Initialize regime parameters
        self._initialize_regime_parameters()
        
        # Default params (will be adjusted based on market regime)
        self.params = {
            # Liquidity influence factors - more conservative baseline
            "l2_weight": 0.6,
            "l3_weight": 0.4,
            "min_signal_score": 20.0,  # Higher threshold for more selective entries
            "cluster_proximity_max": 2.0,
            # Risk and capital management
            "risk_percentage": 0.005,
            "max_balance_risk": 0.75,  # Reduced from 0.80
            "use_fixed_risk": False,
            # Stop loss and take profit settings
            "use_atr_for_stops": False,
            "sl_factor": 1.0,
            "tp_factor": 1.0,
            "base_sl_percent": base_sl_percent,
            "base_tp_percent": base_tp_percent,
            "quality_score_adjustment": False,
            "sl_percent": 1.0,
            "tp_percent": 1.0,
            "trail_start": 0.60, ### TURN OFF FOR TESTING SO 1.0
            "breakeven_point": 0.20,
            "trail_step": 0.4,
            "trail_multiplier": 0.5,
            # Order execution parameters
            "chase_ticks": 3,
            "chase_count": 5,
            "tick_size": 0.5,
            # Enhanced L2/L3 analysis settings
            "anomaly_threshold": 3.0,
            "imbalance_threshold": 2.5,
            "order_flow_weight": 0.5,
            "orderbook_pattern_weight": 0.4,
            "trend_weight": 0.3,
            # Adaptive parameters
            "adapt_thresholds": True,
            "adapt_timeframe": True,
            # New parameters for advanced filtering
            "liquidity_confidence_threshold": 0.7,  # Minimum confidence for liquidity levels
            "min_persistence": 0.4,  # Minimum persistence for liquidity levels
            "false_sweep_filter": True,  # Enable filtering of potential false sweeps
            "order_block_detection": True,  # Enable detection of order blocks
            "mean_reversion_filter": True,  # Filter for mean reversion scenarios
            "dynamic_execution_mode": True,  # Dynamically adjust execution based on market conditions
            "profit_taking_acceleration": True,  # Accelerate profit taking after significant moves
            # Time-based parameters
            "trade_session_filter": True,  # Filter trades based on trading sessions
            "session_filter_weights": {
                "asia": 0.8,  # Multiplier for signal score during Asian session
                "europe": 1.0,  # Multiplier for signal score during European session
                "us": 0.9,  # Multiplier for signal score during US session
            },
            # Performance-based adaptation
            "performance_adaptation": True,  # Adjust parameters based on performance
            "adaptation_lookback": 20,  # Number of trades to consider for adaptation
            "min_trades_for_adaptation": 10,  # Minimum trades before adaptation
            # Advanced position sizing
            "kelly_fraction": 0.3,  # Fraction of Kelly criterion to use
            "use_kelly": False,  # Whether to use Kelly criterion
            # Mean reversion parameters
            "deviation_threshold": 2.0,  # Standard deviations from mean for mean reversion
            "mean_period": 20,  # Period for calculating mean
            # Pattern recognition
            "enable_harmonic_patterns": True,  # Enable harmonic pattern recognition
            "harmonic_pattern_weight": 0.3,  # Weight for harmonic patterns in decision
            # Machine learning models
            "use_ml_models": True,  # Enable ML model predictions
            "ml_prediction_weight": 0.3,  # Weight for ML predictions in decision
            "retraining_frequency": 100,  # Retrain ML models every N trades
            # Stop loss strategies
            "dynamic_stop_placement": True,  # Dynamically place stops based on market noise
            "noise_multiplier": 1.5,  # Multiplier for ATR to determine noise level
            # New trade management settings for high-frequency trading
            "quick_exit_on_failure": True,  # Exit quickly if trade doesn't perform
            "quick_exit_threshold": 0.3,  # Exit if reached this % of stop loss in wrong direction
            "quick_exit_time": 20,  # Seconds to wait before quick exit evaluation
            # Anti-manipulation measures
            "detect_stop_hunting": True,  # Detect and avoid potential stop hunting
            "stop_displacement": 0.15,  # Random displacement factor for stop loss
            # Volume-based adjustments
            "volume_confirmation": True,  # Require volume confirmation for entries
            "min_volume_percentile": 60,  # Minimum volume percentile for confirmation
            # Gradient boosting for quality scoring
            "use_gradient_boosting": True,  # Use gradient boosting for signal quality
            "quality_threshold": 0.5,  # Minimum quality score (0-1)
        }

        # Models for advanced analysis
        self.models = {
            "anomaly_detector": None,
            "quality_predictor": None,
            "regime_classifier": None,
            "pattern_recognizer": None,
        }

        # Exchange configuration - with enhanced credential handling
        self.wallet_address = wallet_address
        self.private_key = private_key

        # Debug credentials
        if debug_mode:
            print(f"\n=== CREDENTIAL DEBUG INFO ===")
            print(f"Wallet address set: {'Yes' if self.wallet_address else 'No'}")
            print(f"Private key set: {'Yes' if self.private_key else 'No'}")
            if self.private_key:
                print(f"Private key length: {len(self.private_key)}")

        # Store the provided exchange instance if available
        if exchange:
            self.exchange = exchange
            logger.info("Using provided exchange instance")
        else:
            # Create exchange with explicit parameters as before
            self.exchange = ccxt.hyperliquid({
                "walletAddress": self.wallet_address,
                "privateKey": self.private_key,
                "options": {"defaultMarketSlippagePercentage": 5.0},
            })

        # Initialize exchange and models
        if not exchange:  # Only initialize exchange if we created it ourselves
            self._initialize_exchange()
        self._initialize_models()
        self._load_trade_history()
        
        # Print initialization status
        logger.info(f"SignalBot initialized for {self.symbol} with {self.timeframe} timeframe")
        logger.info(f"Machine learning models enabled: {self.params['use_ml_models']}")
        logger.info(f"Pattern recognition enabled: {self.params['enable_harmonic_patterns']}")
        logger.info(f"Market regime adaptation enabled: {self.params['adapt_thresholds']}")

    def _initialize_exchange(self):
        """Configures the exchange and initial settings"""
        try:
            # Check if we're using a pre-configured exchange from ExecutionBot
            if hasattr(self, 'exchange') and self.exchange:
                # Exchange already configured, just verify markets
                self.exchange.load_markets()
                logger.info(f"Using pre-configured exchange. {self.symbol} available: {self.symbol in self.exchange.markets}")
                return
                
            # Original exchange initialization code for when not provided by ExecutionBot
            self.exchange.load_markets()
            logger.info(f"Markets loaded. {self.symbol} available: {self.symbol in self.exchange.markets}")
            
            # Skip leverage and margin mode settings - let ExecutionBot handle that
            # Only continue with other initialization as needed
            logger.info(f"Exchange initialized successfully")

        except Exception as e:
            logger.error(f"Error in exchange initialization: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _initialize_regime_parameters(self):
        """Initialize parameter sets for different market regimes"""
        # Default regime parameters from the original code
        self.regime_params = {
            "low_volatility": {
                "l2_weight": 0.8,
                "l3_weight": 0.5,
                "min_signal_score": 50.0,
                "cluster_proximity_max": 1.0,
                "sl_factor": 0.75,
                "tp_factor": 0.5,
                "trail_start": 0.3, 
                "breakeven_point": 0.2,
                "trail_step": 0.4,
                "imbalance_threshold": 2.5,
                "mean_reversion_filter": True,  # Enable mean reversion filtering
                "deviation_threshold": 1.5,     # Lower deviation threshold for mean reversion
            },
            "moderate_volatility": {
                "l2_weight": 0.7,
                "l3_weight": 0.5,
                "min_signal_score": 50.0,
                "cluster_proximity_max": 1.5,
                "sl_factor": 0.65,
                "tp_factor": 0.5,
                "trail_start": 0.65, ### TURN OFF FOR TESTING SO 1.0
                "breakeven_point": 0.2,
                "imbalance_threshold": 3.0,
                "orderbook_pattern_weight": 0.6,  # Increase orderbook pattern weight	
                "order_flow_weight": 0.6,  # Increase order flow weight
            },
            "high_volatility": {
                "l2_weight": 0.6,
                "l3_weight": 0.6,
                "min_signal_score": 50.0,
                "cluster_proximity_max": 2.0,
                "sl_factor": 0.65,
                "tp_factor": 0.5,
                "trail_start": 0.4, ### TURN OFF FOR TESTING SO 1.0
                "breakeven_point": 0.1,
                "quality_threshold": 0.6,  # Increase quality threshold for high volatility
                "false_sweep_filter": True,  # Enable filtering of potential false sweeps
                "stop_hunting_detection": True,  # Enable detection of stop hunting
            },
        }
        
        # Add new parameter sets for trend-based regimes
        self.regime_params.update({
            "strong_uptrend": {
                "l2_weight": 0.5,
                "l3_weight": 0.4,
                "min_signal_score": 60.0,  # Lower threshold to enter more easily during uptrend
                "cluster_proximity_max": 2.0,
                "sl_factor": 0.75,  # Wider stops in trending market
                "tp_factor": 0.5,  # Larger targets in trending market
                "trail_start": 0.4, ### TURN OFF FOR TESTING SO 1.0
                "breakeven_point": 0.15,
                "trend_weight": 0.5,  # Increase trend importance
                "mean_reversion_filter": False,  # Disable mean reversion during strong trend
                "orderbook_scaling_factor": 1.2,  # Increase orderbook influence in uptrend
            },
            "strong_downtrend": {
                "l2_weight": 0.5,
                "l3_weight": 0.4,
                "min_signal_score": 45.0,  # Lower threshold to enter more easily during downtrend
                "cluster_proximity_max": 2.0,
                "sl_factor": 0.75,  # Wider stops in trending market
                "tp_factor": 0.5,  # Larger targets in trending market
                "trail_start": 0.6, ### TURN OFF FOR TESTING SO 1.0
                "breakeven_point": 0.15,
                "trend_weight": 0.5,  # Increase trend importance
                "mean_reversion_filter": False,  # Disable mean reversion during strong trend
                "orderbook_scaling_factor": 1.2,  # Increase orderbook influence in downtrend
            },
            "ranging": {
                "l2_weight": 0.7,
                "l3_weight": 0.3,
                "min_signal_score": 40.0,  # Higher threshold for mean reversion signals
                "cluster_proximity_max": 1.8,
                "sl_factor": 0.4,  # Tighter stops in ranging market
                "tp_factor": 0.5,   # Smaller targets in ranging market
                "trail_start": 0.5, ### TURN OFF FOR TESTING SO 1.0
                "breakeven_point": 0.15,
                "mean_reversion_filter": True,  # Enable mean reversion filtering
                "deviation_threshold": 1.5,     # Lower deviation threshold for mean reversion
            },
            "neutral": {
                # Use the same as moderate_volatility for neutral
                "l2_weight": 0.6,
                "l3_weight": 0.4,
                "min_signal_score": 50.0,
                "cluster_proximity_max": 2.0,
                "sl_factor": 0.65,
                "tp_factor": 0.5,
                "trail_start": 0.5, ### TURN OFF FOR TESTING SO 1.0
                "breakeven_point": 0.2,
            }
        })
        
        # Log initialization
        logger.info(f"Initialized parameters for {len(self.regime_params)} market regimes")

    def _safe_set_leverage(self, leverage, symbol):
        """
        Safely set leverage, handling the case where positions already exist
        
        Returns:
        --------
        bool: Whether the leverage was successfully set
        """
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "open position" in error_msg or "cannot switch" in error_msg:
                # This is expected if positions exist - log and continue
                logger.info(f"Could not set leverage to {leverage}x: existing positions detected")
                
                # Try to get current leverage from positions
                try:
                    positions = self.exchange.fetch_positions([symbol])
                    for position in positions:
                        if float(position.get('contracts', 0)) != 0:
                            current_leverage = position.get('leverage', 'unknown')
                            logger.info(f"Using existing leverage: {current_leverage}x")
                            break
                except:
                    pass
                
                return False
            else:
                # Unexpected error
                logger.error(f"Error setting leverage: {e}")
                return False

    def _sync_position_state(self):  ###################################### CHECK IF WE STILL NEED THIS OR CHANGE IT ############################
        """
        Synchronize the current position state with the exchange
        """
        try:
            # Fetch current open positions
            positions = self.exchange.fetch_positions([self.symbol])
            
            # Check if there are any open positions
            open_positions = [pos for pos in positions if float(pos.get('contracts', 0)) != 0]
            
            if open_positions:
                # Log information about existing positions
                for position in open_positions:
                    logger.info(f"Existing position found: {position.get('side', 'Unknown')} " +
                                f"Size: {position.get('contracts', 0)} " +
                                f"Entry Price: {position.get('entryPrice', 'N/A')}")
                    
                    # Try to get position opening time from timestamp if available
                    if 'timestamp' in position:
                        try:
                            timestamp = position.get('timestamp')
                            if isinstance(timestamp, int):
                                self.position_opened_time = datetime.fromtimestamp(timestamp / 1000)
                                logger.info(f"Position opened at {self.position_opened_time}")
                            else:
                                # If position exists but no timestamp, estimate it as 1 hour ago
                                self.position_opened_time = datetime.now() - timedelta(hours=1)
                                logger.info(f"Position timestamp not available, estimating age")
                        except Exception as e:
                            logger.warning(f"Error parsing position timestamp: {e}")
                            self.position_opened_time = datetime.now() - timedelta(hours=1)
                    else:
                        # No timestamp available, assume position is not brand new
                        self.position_opened_time = datetime.now() - timedelta(hours=1)
                        logger.info(f"Position age unknown, estimating as 1 hour")
                    
                    # Optionally, you can set an active trade state if needed
                    if len(open_positions) > 0:
                        first_position = open_positions[0]
                        self.active_trade = {
                            'side': first_position.get('side', '').lower(),
                            'entry_price': float(first_position.get('entryPrice', 0)),
                            'quantity': float(first_position.get('contracts', 0)),
                            'timestamp': self.position_opened_time
                        }
                else:
                    # No open positions
                    self.active_trade = None
                    self.position_opened_time = None
                    logger.info("No existing positions found")
                
        except Exception as e:
            logger.error(f"Error synchronizing position state: {e}")
            # Ensure active_trade is reset in case of any error
            self.active_trade = None

    def update_parameters(self, new_params):
        """Updates strategy parameters dynamically during runtime"""
        try:
            for key, value in new_params.items():
                if key in self.params:
                    old_value = self.params[key]
                    self.params[key] = value
                    logger.info(f"Parameter {key} updated: {old_value} -> {value}")
                else:
                    logger.warning(f"Unknown parameter: {key}")

            return {"success": True, "updated_params": list(new_params.keys())}

        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return {"success": False, "reason": str(e)}
    
    
    ##################################################################################################################################################
    ########################################################### Data Retrieval and Processing ########################################################
    ##################################################################################################################################################

    def fetch_data(self, limit=200, force_timeframe_check=False):
        """Retrieves historical OHLCV data with controlled timeframe adaptation"""
        try:
            # Default timeframe
            timeframe = self.timeframe
            
            # Track last timeframe check time if not already tracking
            if not hasattr(self, 'last_timeframe_check'):
                self.last_timeframe_check = datetime.now() - timedelta(minutes=30)  # Force initial check
                
            # Adaptive timeframe if enabled and it's time to check again
            time_since_last_check = (datetime.now() - self.last_timeframe_check).total_seconds() / 60
            timeframe_check_interval = 10  # Check every 15 minutes
            
            if (force_timeframe_check or time_since_last_check >= timeframe_check_interval) and \
            self.params.get("adapt_timeframe", False):
                
                # Run timeframe analysis
                self.last_timeframe_check = datetime.now()
                
                # Get market snapshot to detect volatility
                if hasattr(self, 'data') and self.data is not None:
                    # Try to infer volatility from most recent ATR data
                    try:
                        if 'atr_percent' in self.data.columns:
                            recent_atr_pct = self.data['atr_percent'].iloc[-1]
                            
                            # Log current volatility for reference
                            logger.info(f"Current ATR: {recent_atr_pct:.2f}% (Checking timeframe suitability)")
                            
                            if recent_atr_pct > 0.9:  # High volatility
                                if self.timeframe == "5m":
                                    timeframe = "1m"
                                elif self.timeframe == "15m":
                                    timeframe = "5m"
                                logger.info(f"Adjusting timeframe to {timeframe} due to high ATR ({recent_atr_pct:.2f}%)")
                            elif recent_atr_pct < 0.4:  # Low volatility
                                if self.timeframe == "5m":
                                    timeframe = "15m"
                                elif self.timeframe == "1m":
                                    timeframe = "5m"
                                logger.info(f"Adjusting timeframe to {timeframe} due to low ATR ({recent_atr_pct:.2f}%)")
                            else:
                                # Volatility is moderate, revert to original timeframe if different
                                if timeframe != self.timeframe:
                                    timeframe = self.timeframe
                                    logger.info(f"Reverting to original timeframe {timeframe} (ATR: {recent_atr_pct:.2f}%)")
                    except Exception as e:
                        logger.warning(f"Error in volatility-based timeframe adaptation: {e}")
            
            # Fetch OHLCV data with selected timeframe
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol, timeframe=timeframe, limit=limit
            )

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            if self.debug_mode:
                logger.info(f"OHLCV data retrieved: {len(df)} candles with timeframe {timeframe}")
                logger.info(
                    f"Statistics: mean={df['close'].mean():.2f}, min={df['low'].min():.2f}, max={df['high'].max():.2f}"
                )

            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error in retrieving OHLCV data: {e}")
            return None

    ####### Testing new fetch_data method with adaptive timeframe #######
    #def fetch_data(self, limit=1000):
        """Retrieves historical OHLCV data with automatic timeframe adaptation"""
        try:
            # Default timeframe
            timeframe = self.timeframe
            
            # Adaptive timeframe if enabled
            if hasattr(self, 'params') and self.params.get("adapt_timeframe", False):
                # Get market snapshot to detect volatility
                if hasattr(self, 'market_snapshot') and self.market_snapshot and "regimes" in self.market_snapshot:
                    vol_regime = self.market_snapshot["regimes"].get("volatility", "moderate")
                    if vol_regime == "high":
                        # Use shorter timeframe in high volatility
                        if self.timeframe == "5m":
                            timeframe = "1m"
                        elif self.timeframe == "15m":
                            timeframe = "5m"
                        logger.info(f"Adjusting timeframe to {timeframe} due to high volatility")
                    elif vol_regime == "low":
                        # Use longer timeframe in low volatility
                        if self.timeframe == "5m":
                            timeframe = "15m"
                        elif self.timeframe == "1m":
                            timeframe = "5m"
                        logger.info(f"Adjusting timeframe to {timeframe} due to low volatility")
                elif hasattr(self, 'data') and self.data is not None:
                    # Try to infer volatility from most recent ATR data
                    try:
                        if 'atr_percent' in self.data.columns:
                            recent_atr_pct = self.data['atr_percent'].iloc[-1]
                            if recent_atr_pct > 1.2:  # High volatility
                                if self.timeframe == "5m":
                                    timeframe = "1m"
                                elif self.timeframe == "15m":
                                    timeframe = "5m"
                                logger.info(f"Adjusting timeframe to {timeframe} due to high ATR ({recent_atr_pct:.2f}%)")
                            elif recent_atr_pct < 0.5:  # Low volatility
                                if self.timeframe == "5m":
                                    timeframe = "15m"
                                elif self.timeframe == "1m":
                                    timeframe = "5m"
                                logger.info(f"Adjusting timeframe to {timeframe} due to low ATR ({recent_atr_pct:.2f}%)")
                    except Exception as e:
                        logger.warning(f"Error in volatility-based timeframe adaptation: {e}")

            # Fetch OHLCV data with selected timeframe
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol, timeframe=timeframe, limit=limit
            )

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            if self.debug_mode:
                logger.info(f"OHLCV data retrieved: {len(df)} candles with timeframe {timeframe}")
                logger.info(
                    f"Statistics: mean={df['close'].mean():.2f}, min={df['low'].min():.2f}, max={df['high'].max():.2f}"
                )

            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error in retrieving OHLCV data: {e}")
            return None

    def calculate_indicators(self):
        """Calculates technical indicators for market analysis"""
        if self.data is None:
            self.data = self.fetch_data()

        try:
            # Create copy of DataFrame
            df = self.data.copy()

            # Calculate HL2 (average between high and low)
            df["hl2"] = (df["high"] + df["low"]) / 2

            # Calculate RSI and ADX to evaluate trend and momentum
            df["rsi"] = df.ta.rsi(close="close", length=14)
            df["adx"] = df.ta.adx(high="high", low="low", close="close", length=14)["ADX_14"]

            # ATR for volatility and risk management
            df["atr"] = df.ta.atr(high="high", low="low", close="close", length=14)

            # EMA for trend
            df["ema50"] = df.ta.ema(close="close", length=50)
            df["ema200"] = df.ta.ema(close="close", length=200)

            # MACD for momentum analysis
            macd_result = df.ta.macd(close="close")
            df["macd"] = macd_result["MACD_12_26_9"]
            df["macd_signal"] = macd_result["MACDs_12_26_9"]
            df["macd_hist"] = macd_result["MACDh_12_26_9"]

            # Bollinger Bands for volatility assessment
            bbands = df.ta.bbands(close="close", length=20, std=2)
            df["bb_upper"] = bbands["BBU_20_2.0"]
            df["bb_middle"] = bbands["BBM_20_2.0"]
            df["bb_lower"] = bbands["BBL_20_2.0"]

            # Stochastic Oscillator to identify overbought/oversold
            stoch = df.ta.stoch(high="high", low="low", close="close", k=14, d=3, smooth_k=3)
            df["slowk"] = stoch["STOCHk_14_3_3"]
            df["slowd"] = stoch["STOCHd_14_3_3"]

            # Calculate EMA slope to determine trend strength
            if len(df) > 10:
                df["ema50_slope"] = (df["ema50"] - df["ema50"].shift(5)) / 5
                df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / 10

            # Add volume-based indicators
            if "volume" in df.columns:
                # On-Balance Volume (OBV)
                df["obv"] = df.ta.obv(close="close", volume="volume")
                # Chaikin Money Flow
                df["cmf"] = df.ta.cmf(high="high", low="low", close="close", volume="volume", length=10)
                # Money Flow Index
                df["mfi"] = df.ta.mfi(high="high", low="low", close="close", volume="volume", length=14)

                # Volume moving averages
                df["volume_sma5"] = df["volume"].rolling(5).mean()
                df["volume_sma20"] = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma20"]

            # Add volatility indicators for adaptive parameters
            df["atr_percent"] = df["atr"] / df["close"] * 100

            # Add Heikin-Ashi candles for trend assessment
            df["ha_open"] = 0.0
            df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

            for i in range(1, len(df)):
                df.iloc[i, df.columns.get_loc("ha_open")] = (
                    df.iloc[i - 1, df.columns.get_loc("ha_open")]
                    + df.iloc[i - 1, df.columns.get_loc("ha_close")]
                ) / 2

            df["ha_high"] = df[["high", "ha_open", "ha_close"]].max(axis=1)
            df["ha_low"] = df[["low", "ha_open", "ha_close"]].min(axis=1)

            # Additional indicators for pattern detection
            # Fisher Transform for market turning points
            df["high_9"] = df["high"].rolling(9).max()
            df["low_9"] = df["low"].rolling(9).min()
            df["price_location"] = (df["close"] - df["low_9"]) / (
                df["high_9"] - df["low_9"]
            )

            # Add normalized price momentum
            df["momentum"] = (
                (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100
            )

            # Directional indicators for trend quality
            dmi = df.ta.adx(high="high", low="low", close="close", length=14)
            df["plus_di"] = dmi["DMP_14"]
            df["minus_di"] = dmi["DMN_14"]

            # Advanced mean reversion indicators
            df["zscore"] = (df["close"] - df["close"].rolling(20).mean()) / df[
                "close"
            ].rolling(20).std()

            # Add market regime detection
            df["regime"] = "neutral"
            df.loc[(df["atr_percent"] < 0.5), "regime"] = "low_volatility"
            df.loc[(df["atr_percent"] > 1.2), "regime"] = "high_volatility"
            df.loc[(df["adx"] > 25) & (df["plus_di"] > df["minus_di"]), "regime"] = (
                "strong_uptrend"
            )
            df.loc[(df["adx"] > 25) & (df["plus_di"] < df["minus_di"]), "regime"] = (
                "strong_downtrend"
            )
            df.loc[(df["adx"] < 20), "regime"] = "ranging"

            # Update the main DataFrame
            self.data = df
            logger.info("Technical indicators calculated successfully")

            # Update market regime
            current_regime = df["regime"].iloc[-1]
            if current_regime in self.regime_params:
                # Only update parameters if regime changed
                if self.market_regime != current_regime:
                    logger.info(
                        f"Market regime changed from {self.market_regime} to {current_regime}, updating parameters"
                    )
                    self._adapt_parameters_to_regime(current_regime)
                    self.market_regime = current_regime

            return df
        except Exception as e:
            logger.error(f"Error in calculating indicators: {e}")
            return None

    def fetch_orderbook(self, depth=200):
        """Retrieves and analyzes the orderbook with in-depth L2/L3 analysis"""
        try:
            # Retrieve complete orderbook
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=depth)

            # Orderbook timestamp
            orderbook["timestamp"] = datetime.now()

            # Calculate basic metrics
            bid_volume = sum([order[1] for order in orderbook["bids"]])
            ask_volume = sum([order[1] for order in orderbook["asks"]])

            # Calculate bid/ask ratio
            bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0

            # Calculate spread and mid price
            best_bid = orderbook["bids"][0][0] if orderbook["bids"] else 0
            best_ask = orderbook["asks"][0][0] if orderbook["asks"] else 0
            spread = best_ask - best_bid
            spread_percent = (spread / best_bid) * 100 if best_bid > 0 else 0
            mid_price = (best_bid + best_ask) / 2
            
            logger.info(f"Orderbook Updated: {len(orderbook['bids'])} bid e {len(orderbook['asks'])} ask")
            logger.info(f"Spread: {spread_percent:.3f}%, Average price: {mid_price:.2f}")
            logger.info(f"Bid volume: {bid_volume:.2f}, Ask volume: {ask_volume:.2f}, Ratio: {bid_ask_ratio:.2f}")

            # ADVANCED L2 ANALYSIS
            # Generate liquidity heatmap
            l2_heatmap = self._generate_l2_heatmap(orderbook)

            # Analysis of volume distribution
            bid_distribution = self._analyze_volume_distribution(orderbook["bids"])
            ask_distribution = self._analyze_volume_distribution(orderbook["asks"])

            # Identification of liquidity barriers and equilibrium zones
            liquidity_barriers = self._identify_liquidity_barriers(orderbook)

            # Detection of significant imbalances
            imbalance_zones = self._detect_imbalance_zones(
                orderbook, threshold=self.params["imbalance_threshold"]
            )

            # Advanced identification of liquidity levels
            bid_liquidity_levels = self._identify_advanced_liquidity_levels(
                orderbook["bids"]
            )
            ask_liquidity_levels = self._identify_advanced_liquidity_levels(
                orderbook["asks"]
            )

            # Advanced detection of order walls (large orders)
            bid_walls = self._detect_advanced_order_walls(orderbook["bids"])
            ask_walls = self._detect_advanced_order_walls(orderbook["asks"])

            # Analysis of liquidity changes over time
            orderbook_with_raw = {"raw": orderbook, "timestamp": orderbook["timestamp"]}
            liquidity_changes = self._analyze_liquidity_changes(orderbook_with_raw)

            # L3 ANALYSIS - Advanced interpretation of orders
            # Analysis of order "Footprint"
            footprint = self._analyze_order_footprint(orderbook)

            # Detect orderbook patterns
            orderbook_patterns = self._detect_orderbook_patterns(orderbook)

            # Anomaly detection in orderbook structure
            orderbook_anomalies = self._detect_orderbook_anomalies(orderbook)

            # NEW: Detect potential market manipulations
            manipulation_signals = self._detect_manipulation_signals(orderbook)

            # NEW: Cluster analysis for order flow
            order_clusters = self._analyze_order_clusters(orderbook)

            # NEW: Detect stop hunting zones
            stop_hunting_zones = self._detect_stop_hunting_zones(orderbook)

            # NEW: Identify spoofing patterns using ML
            spoofing_patterns = self._identify_spoofing_patterns(orderbook)

            # Build an enriched orderbook with L2/L3 analysis
            enriched_orderbook = {
                "raw": orderbook,
                "timestamp": orderbook["timestamp"],
                "metrics": {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "spread_percent": spread_percent,
                    "mid_price": mid_price,
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "bid_ask_ratio": bid_ask_ratio,
                },
                "liquidity": {
                    "bid_levels": bid_liquidity_levels,
                    "ask_levels": ask_liquidity_levels,
                    "bid_walls": bid_walls,
                    "ask_walls": ask_walls,
                    "heatmap": l2_heatmap,
                    "changes": liquidity_changes,
                    "barriers": liquidity_barriers,
                    "imbalance": imbalance_zones,
                    "bid_distribution": bid_distribution,
                    "ask_distribution": ask_distribution,
                    "clusters": order_clusters,
                },
                "l3_analysis": {
                    "footprint": footprint,
                    "patterns": orderbook_patterns,
                    "anomalies": orderbook_anomalies,
                    "manipulation": manipulation_signals,
                    "stop_hunting": stop_hunting_zones,
                    "spoofing": spoofing_patterns,
                },
            }

            # Store the orderbook for historical analysis
            self.orderbook_history.append(enriched_orderbook)
            self.orderbook_data = enriched_orderbook
            self.l2_heatmap = l2_heatmap

            # Update the global market snapshot
            self._update_market_snapshot()

            # Log key metrics
            if self.debug_mode:
                logger.info(
                    f"Orderbook updated - Spread: {spread_percent:.3f}%, Ratio B/A: {bid_ask_ratio:.2f}"
                )
                if bid_walls:
                    logger.info(f"Buy liquidity walls detected: {len(bid_walls)}")
                if ask_walls:
                    logger.info(f"Sell liquidity walls detected: {len(ask_walls)}")

                # Log significant liquidity barriers
                if liquidity_barriers and liquidity_barriers.get(
                    "significant_barriers", False
                ):
                    most_sig = liquidity_barriers.get("most_significant_barrier", {})
                    logger.info(
                        f"Significant liquidity barrier: {most_sig.get('type', '')} @ {most_sig.get('price', 0):.2f} (strength: {most_sig.get('strength', 0):.1f}/10)"
                    )

                # Log imbalance zones
                if imbalance_zones and len(imbalance_zones.get("zones", [])) > 0:
                    strongest_zone = imbalance_zones["zones"][
                        0
                    ]  # The first is the strongest
                    logger.info(
                        f"Imbalance zone detected: {strongest_zone['type']} {strongest_zone['price_range'][0]:.2f}-{strongest_zone['price_range'][1]:.2f} (ratio: {strongest_zone['ratio']:.2f})"
                    )

                # Log orderbook patterns and anomalies
                if orderbook_patterns and orderbook_patterns.get(
                    "detected_patterns", []
                ):
                    logger.info(
                        f"Orderbook patterns detected: {', '.join([p['name'] for p in orderbook_patterns['detected_patterns']])}"
                    )

                if orderbook_anomalies and orderbook_anomalies.get("anomalies", []):
                    logger.info(
                        f"Orderbook anomalies detected: {len(orderbook_anomalies['anomalies'])}"
                    )

                # Log manipulation signals if any
                if manipulation_signals and manipulation_signals.get("detected", False):
                    logger.warning(
                        f"Potential market manipulation detected: {manipulation_signals['type']}"
                    )

            return enriched_orderbook

        except Exception as e:
            logger.error(f"Error in retrieving orderbook: {e}")
            return None

    def fetch_trades(self, limit=200):
        """Fetch recent trades for the symbol"""
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            
            # Process trades for analysis
            processed_trades = []
            for trade in trades:
                processed_trade = {
                    'timestamp': trade['timestamp'],
                    'datetime': trade['datetime'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side'],
                    'cost': trade['cost'] if 'cost' in trade else trade['price'] * trade['amount']
                }
                processed_trades.append(processed_trade)
            
            # Add to trades history
            for trade in processed_trades:
                self.trades_history.append(trade)
            
            # Update market snapshot with recent trades
            self._update_market_snapshot(trades=processed_trades)
            
            return processed_trades
        except Exception as e:
            logger.error(f"Error in fetching trades: {e}")
            return []
    
    def _update_market_snapshot(self, orderbook=None, trades=None):
        """Update market snapshot with latest orderbook and trades data"""
        try:
            if not self.market_snapshot:
                self.market_snapshot = {}
            
            current_time = datetime.now()
            
            if orderbook:
                self.market_snapshot['orderbook'] = orderbook
                self.market_snapshot['orderbook_time'] = current_time
            
            if trades:
                self.market_snapshot['recent_trades'] = trades
                self.market_snapshot['trades_time'] = current_time
                
                # Calculate trade stats
                buy_volume = sum([t['amount'] for t in trades if t['side'] == 'buy'])
                sell_volume = sum([t['amount'] for t in trades if t['side'] == 'sell'])
                
                self.market_snapshot['buy_volume'] = buy_volume
                self.market_snapshot['sell_volume'] = sell_volume
                self.market_snapshot['volume_ratio'] = buy_volume / sell_volume if sell_volume > 0 else 1.0
            
            # Update snapshot timestamp
            self.market_snapshot['last_update'] = current_time
            
            return self.market_snapshot
        except Exception as e:
            logger.error(f"Error in updating market snapshot: {e}")
            return None

    ##################################################################################################################################################
    ####################################################### Market Regime and Parameter Adaptation ###################################################
    ##################################################################################################################################################
    
    def _adapt_parameters_to_regime(self, regime):
        """Adapt parameters based on current market regime with enhanced logging"""
        try:
            if regime not in self.regime_params:
                logger.warning(f"Unknown regime: {regime}, using default parameters")
                regime = "neutral"  # Fallback to neutral regime
                
            # Get regime-specific parameters
            regime_params = self.regime_params[regime]
            
            # Log previous key parameters before update
            old_min_signal_score = self.params.get("min_signal_score", "N/A")
            old_sl_factor = self.params.get("sl_factor", "N/A")
            old_tp_factor = self.params.get("tp_factor", "N/A")

            # Update current parameters with regime-specific ones
            for key, value in regime_params.items():
                self.params[key] = value

            # Log key parameter changes
            logger.info(f"Parameters adapted to '{regime}' regime")
            logger.info(f"  Signal threshold: {old_min_signal_score} -> {self.params['min_signal_score']}")
            logger.info(f"  Stop loss factor: {old_sl_factor} -> {self.params['sl_factor']}")
            logger.info(f"  Take profit factor: {old_tp_factor} -> {self.params['tp_factor']}")

            # Additional adaptations based on current performance
            if (
                self.params["performance_adaptation"]
                and len(self.trade_history) >= self.params["min_trades_for_adaptation"]
            ):
                recent_trades = self.trade_history[
                    -self.params["adaptation_lookback"] :
                ]
                win_rate = sum(
                    1 for t in recent_trades if t.get("profit_pct", 0) > 0
                ) / len(recent_trades)

                # Calculate average profit/loss ratio
                if len(recent_trades) > 0:
                    profits = [t.get("profit_pct", 0) for t in recent_trades if t.get("profit_pct", 0) > 0]
                    losses = [abs(t.get("profit_pct", 0)) for t in recent_trades if t.get("profit_pct", 0) < 0]
                    
                    avg_profit = sum(profits) / len(profits) if profits else 0
                    avg_loss = sum(losses) / len(losses) if losses else 0
                    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
                    
                    logger.info(f"Recent performance: Win rate: {win_rate:.2f}, P/L ratio: {profit_loss_ratio:.2f}")

                # If win rate is low, increase selectivity
                if win_rate < 0.4:
                    self.params["min_signal_score"] += 0.2
                    logger.info(
                        f"Low win rate detected ({win_rate:.2f}), increasing min_signal_score to {self.params['min_signal_score']}"
                    )
                # If win rate is high, we can be less selective
                elif win_rate > 0.6:
                    self.params["min_signal_score"] = max(
                        8.0, self.params["min_signal_score"] - 0.1
                    )
                    logger.info(
                        f"High win rate detected ({win_rate:.2f}), decreasing min_signal_score to {self.params['min_signal_score']}"
                    )

        except Exception as e:
            logger.error(f"Error adapting parameters to regime: {e}")

    def adjust_signal_weights(self, weights_config=None):
        """
        Adjust signal generation weights to prioritize different data sources
        
        Parameters:
        -----------
        weights_config : dict
            Configuration of weights to apply
        """
        if weights_config is None:
            # Default configuration that prioritizes orderbook data
            weights_config = {
                "orderbook_multiplier": 1.3,     # Multiply orderbook signals by this factor
                "technical_multiplier": 0.7,     # Multiply technical signals by this factor
                "l2_weight": 0.7,                # L2 orderbook data weight
                "l3_weight": 0.5,                # L3 analysis weight
                "order_flow_weight": 0.6,        # Order flow analysis weight
                "orderbook_pattern_weight": 0.5, # Orderbook pattern weight
                "trend_weight": 0.2,             # Technical trend weight
                "quality_threshold": 0.5,        # Signal quality threshold (0-1)
                "min_signal_score": 20.0          # Minimum signal score for generation
            }
        
        logger.info("Adjusting signal weights to prioritize orderbook data")
        
        # Apply to parameters
        self.params["l2_weight"] = weights_config.get("l2_weight", self.params.get("l2_weight"))
        self.params["l3_weight"] = weights_config.get("l3_weight", self.params.get("l3_weight"))
        self.params["order_flow_weight"] = weights_config.get("order_flow_weight", self.params.get("order_flow_weight"))
        self.params["orderbook_pattern_weight"] = weights_config.get("orderbook_pattern_weight", self.params.get("orderbook_pattern_weight"))
        self.params["trend_weight"] = weights_config.get("trend_weight", self.params.get("trend_weight"))
        self.params["quality_threshold"] = weights_config.get("quality_threshold", self.params.get("quality_threshold"))
        self.params["min_signal_score"] = weights_config.get("min_signal_score", self.params.get("min_signal_score"))
        
        # Store the multipliers for use in signal generation
        self.params["orderbook_multiplier"] = weights_config.get("orderbook_multiplier", 1.0)
        self.params["technical_multiplier"] = weights_config.get("technical_multiplier", 1.0)
        
        # Apply to all regime parameters as well
        for regime in self.regime_params:
            self.regime_params[regime]["l2_weight"] = weights_config.get("l2_weight", self.regime_params[regime].get("l2_weight"))
            self.regime_params[regime]["l3_weight"] = weights_config.get("l3_weight", self.regime_params[regime].get("l3_weight"))
            self.regime_params[regime]["order_flow_weight"] = weights_config.get("order_flow_weight", self.regime_params[regime].get("order_flow_weight"))
            self.regime_params[regime]["orderbook_pattern_weight"] = weights_config.get("orderbook_pattern_weight", self.regime_params[regime].get("orderbook_pattern_weight"))
            self.regime_params[regime]["trend_weight"] = weights_config.get("trend_weight", self.regime_params[regime].get("trend_weight"))
            self.regime_params[regime]["min_signal_score"] = weights_config.get("min_signal_score", self.regime_params[regime].get("min_signal_score"))
        
        logger.info(f"Signal weights adjusted: L2={self.params['l2_weight']}, L3={self.params['l3_weight']}, OrderFlow={self.params['order_flow_weight']}")
        logger.info(f"Signal thresholds: Quality={self.params['quality_threshold']}, MinScore={self.params['min_signal_score']}")
        
        return True

    def _create_regime_classifier(self):
        """Create a classifier for market regimes"""
        try:
            # Simple classifier based on statistical measures
            classifier = {
                "volatility_thresholds": {
                    "low": 0.4,  # Below 0.4% ATR/price is low volatility
                    "high": 0.9,  # Above 0.9% ATR/price is high volatility
                },
                "volume_thresholds": {
                    "low": 0.7,  # Below 70% of average volume is low
                    "high": 1.3,  # Above 130% of average volume is high
                },
                "price_thresholds": {
                    "overbought": 70,  # RSI above 70 is overbought
                    "oversold": 30,  # RSI below 30 is oversold
                },
            }
            return classifier
        except Exception as e:
            logger.error(f"Error creating regime classifier: {e}")
            return None

    def analyze_multi_timeframe(self):
        """Analyze multiple timeframes and synthesize a more stable market regime"""
        try:
            # Store original timeframe
            original_timeframe = self.timeframe
            timeframes = ['1m', '5m', '15m']
            regime_votes = {}
            
            for tf in timeframes:
                # Temporarily set timeframe
                self.timeframe = tf
                # Fetch data without adaptive timeframe adjustment
                ohlcv = self.exchange.fetch_ohlcv(symbol=self.symbol, timeframe=tf, limit=300)
                
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                
                # Calculate basic indicators for regime detection
                df["atr"] = df.ta.atr(high="high", low="low", close="close", length=14)
                df["atr_percent"] = df["atr"] / df["close"] * 100
                dmi = df.ta.adx(high="high", low="low", close="close", length=14)
                df["adx"] = dmi["ADX_14"]
                df["plus_di"] = dmi["DMP_14"]
                df["minus_di"] = dmi["DMN_14"]
                
                # Detect regime
                last_row = df.iloc[-1]
                
                # Determine regime
                regime = "neutral"
                if last_row["atr_percent"] < 0.4:
                    regime = "low_volatility"
                elif last_row["atr_percent"] > 0.9:
                    regime = "high_volatility"
                    
                if last_row["adx"] > 25 and last_row["plus_di"] > last_row["minus_di"]:
                    regime = "strong_uptrend"
                elif last_row["adx"] > 25 and last_row["plus_di"] < last_row["minus_di"]:
                    regime = "strong_downtrend"
                elif last_row["adx"] < 20:
                    regime = "ranging"
                    
                # Add vote with weight (higher timeframes get more weight)
                weight = 1
                if tf == '15m':
                    weight = 2
                elif tf == '1h':
                    weight = 3
                    
                if regime in regime_votes:
                    regime_votes[regime] += weight
                else:
                    regime_votes[regime] = weight
                    
                logger.info(f"Timeframe {tf} regime: {regime}")
                    
            # Restore original timeframe
            self.timeframe = original_timeframe
            
            # Determine the winning regime
            current_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
            logger.info(f"Multi-timeframe analysis result: {current_regime} (votes: {regime_votes})")
            
            # Only update regime if it's clearly different
            if self.market_regime != current_regime:
                logger.info(f"Market regime changed from {self.market_regime} to {current_regime}, updating parameters")
                self._adapt_parameters_to_regime(current_regime)
                self.market_regime = current_regime
                
            return current_regime
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return self.market_regime or "neutral"

    ###### New testing:
    def _enhance_orderbook_sensitivity(self):
        """Adjust sensitivity of orderbook analysis to provide more differentiated values"""
        # Modify these parameters in your bot
        self.params["imbalance_threshold"] = 2.0  # Lower from 3.0 - detect smaller imbalances
        self.params["min_signal_score"] = 20.0  # You can adjust this based on your preference
        
        # Add these parameters
        self.params["orderbook_scaling_factor"] = 1.5  # Increases the range of values
        self.params["significance_threshold"] = 0.8  # Proportion of the mean to consider significant


    ##################################################################################################################################################
    ############################################################### ML Model Management ##############################################################
    ##################################################################################################################################################

    def _initialize_models(self):
        """Initialize all machine learning models for advanced analysis"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # 1. Anomaly Detector (load if exists, create new otherwise)
            if os.path.exists("models/anomaly_detector.pkl"):
                with open("models/anomaly_detector.pkl", "rb") as f:
                    self.models["anomaly_detector"] = pickle.load(f)
                logger.info("Loaded anomaly detection model")
            else:
                # Initialize with default model
                self.models["anomaly_detector"] = IsolationForest(
                    contamination=0.1, random_state=42
                )
                logger.info("Created new anomaly detection model")
            
            # 2. Quality Predictor (for signal quality estimation)
            self._initialize_quality_predictor()
            if self.models["quality_predictor"] is not None:
                logger.info("Quality predictor initialized successfully")
            
            # 3. Pattern Recognizer (for technical pattern detection)
            if os.path.exists("models/pattern_recognizer.pkl"):
                with open("models/pattern_recognizer.pkl", "rb") as f:
                    self.models["pattern_recognizer"] = pickle.load(f)
                logger.info("Loaded pattern recognition model")
            else:
                # Initialize as None - will be created on-demand based on data
                self.models["pattern_recognizer"] = None
                logger.info("Pattern recognizer will be initialized when needed")
            
            # 4. Regime Classifier (for market regime detection)
            self.models["regime_classifier"] = self._create_regime_classifier()
            logger.info("Initialized regime classifier")
            
            # Track model performance and retraining metrics
            self.model_metadata = {
                "samples_since_last_training": 0,
                "retraining_frequency": self.params["retraining_frequency"],
                "last_training_time": None,
                "performance_metrics": {}
            }
            
            # Save initial models to disk
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Set defaults if model initialization fails
            self.models["anomaly_detector"] = IsolationForest(
                contamination=0.1, random_state=42
            )
            self.models["quality_predictor"] = RandomForestRegressor(
                n_estimators=20, random_state=42
            )
            self.models["pattern_recognizer"] = None
            self.models["regime_classifier"] = self._create_regime_classifier()

    def _initialize_quality_predictor(self):
        """Initialize the quality predictor with default values if no historical data exists"""
        try:
            # Check if model exists
            if "quality_predictor" not in self.models or self.models["quality_predictor"] is None:
                from sklearn.ensemble import RandomForestRegressor
                
                # Create a basic model
                self.models["quality_predictor"] = RandomForestRegressor(
                    n_estimators=20, max_depth=5, random_state=42
                )
                
                # Create simple synthetic training data to initially fit the model
                # Features: [rsi, adx, atr_percent, volume_ratio, macd]
                X_synthetic = [
                    [30, 20, 0.5, 1.0, -2],  # Oversold
                    [70, 20, 0.5, 1.0, 2],   # Overbought
                    [45, 30, 1.0, 1.5, 1],   # Strong trend with volume
                    [55, 10, 0.3, 0.7, 0],   # Weak trend, low volatility
                    [60, 25, 0.8, 1.2, 1.5], # Moderate uptrend
                    [40, 25, 0.8, 1.2, -1.5] # Moderate downtrend
                ]
                
                # Target values (simulated quality scores)
                y_synthetic = [6.5, 6.0, 7.5, 4.5, 6.0, 5.5]
                
                # Fit the model with synthetic data
                self.models["quality_predictor"].fit(X_synthetic, y_synthetic)
                
                logger.info("Quality predictor initialized with synthetic data")
                
                # Save model for future use
                self._save_models()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing quality predictor: {e}")
            return False

    def _check_model_retraining(self):
        """Check if models need retraining based on new data"""
        try:
            # Increment samples counter
            if not hasattr(self, 'model_metadata'):
                self.model_metadata = {
                    "samples_since_last_training": 0,
                    "retraining_frequency": self.params.get("retraining_frequency", 100),
                    "last_training_time": None,
                    "performance_metrics": {}
                }
            
            self.model_metadata["samples_since_last_training"] += 1
            
            # Check if retraining is needed
            if self.model_metadata["samples_since_last_training"] >= self.model_metadata["retraining_frequency"]:
                logger.info(f"Retraining models after {self.model_metadata['samples_since_last_training']} samples")
                self._retrain_models()
                self.model_metadata["samples_since_last_training"] = 0
                self.model_metadata["last_training_time"] = datetime.now()
        except Exception as e:
            logger.error(f"Error checking model retraining: {e}")

    def _retrain_models(self):
        """Retrain machine learning models with new data"""
        try:
            # 1. Prepare training data from recent market observations
            if not hasattr(self, 'data') or self.data is None:
                logger.warning("No data available for model retraining")
                return
                
            recent_data = self.data.tail(min(len(self.data), 1000))  # Use last 1000 samples max
            
            # 2. Retrain anomaly detector
            if "anomaly_detector" in self.models and self.models["anomaly_detector"] is not None:
                logger.info("Retraining anomaly detector model")
                
                # Prepare features for anomaly detection
                features = []
                if 'atr_percent' in recent_data.columns and 'volume_ratio' in recent_data.columns:
                    features = recent_data[['atr_percent', 'volume_ratio']].dropna().values
                    
                    # Additional features if available
                    optional_features = ['rsi', 'adx', 'macd', 'obv']
                    available_features = [f for f in optional_features if f in recent_data.columns]
                    
                    if available_features:
                        features = np.column_stack([features, recent_data[available_features].dropna().values])
                
                if len(features) > 10:  # Need minimum samples for meaningful training
                    # Standardize the data
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    # Retrain the model
                    self.models["anomaly_detector"].fit(features_scaled)
                    logger.info(f"Anomaly detector retrained with {len(features)} samples")
                else:
                    logger.warning("Insufficient data for anomaly detector retraining")
            
            # 3. Retrain quality predictor if we have trade history to learn from
            if "quality_predictor" in self.models and len(self.trade_history) >= 20:
                logger.info("Retraining quality predictor model")
                
                # Extract features and target variable from past trades
                feature_columns = ['rsi', 'adx', 'atr_percent', 'volume_ratio', 'macd'] 
                available_features = [f for f in feature_columns if f in recent_data.columns]
                
                if available_features:
                    # Collect training samples
                    X_train = []
                    y_train = []
                    
                    for trade in self.trade_history[-50:]:  # Use last 50 trades max
                        if 'entry_time' not in trade:
                            continue
                            
                        # Find market state at trade entry
                        entry_time = trade['entry_time']
                        
                        # Find closest data point to entry time
                        if isinstance(entry_time, str):
                            entry_time = pd.to_datetime(entry_time)
                        
                        # Find the data point closest to entry time
                        idx = recent_data.index.get_indexer([entry_time], method='nearest')[0]
                        if idx >= 0 and idx < len(recent_data):
                            features = recent_data.iloc[idx][available_features].values
                            profit = trade.get('profit_pct', 0)
                            
                            X_train.append(features)
                            y_train.append(profit)
                    
                    if len(X_train) >= 10:  # Need minimum samples
                        # Standardize features
                        X_train = np.array(X_train)
                        y_train = np.array(y_train)
                        
                        # Retrain the model
                        self.models["quality_predictor"].fit(X_train, y_train)
                        
                        # Test on training data to see if it's learning
                        y_pred = self.models["quality_predictor"].predict(X_train)
                        mse = np.mean((y_pred - y_train) ** 2)
                        
                        logger.info(f"Quality predictor retrained with {len(X_train)} samples. Training MSE: {mse:.4f}")
                        
                        # Save performance metrics
                        self.model_metadata["performance_metrics"]["quality_predictor_mse"] = mse
                    else:
                        logger.warning("Insufficient trade data for quality predictor retraining")
            
            # 4. Save updated models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")

    def _save_models(self):
        """Save all models to disk"""
        try:
            os.makedirs("models", exist_ok=True)
            
            # Save each model that's not None
            for model_name, model in self.models.items():
                if model is not None and model_name != "regime_classifier":  # Skip regime classifier as it's a simple dict
                    model_path = f"models/{model_name}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                    logger.info(f"Saved {model_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    ###### New testing (OLD ML IS AT THE BOTTOM OF THIS METHOD) ######:
    def _predict_signal_quality(self, market_features):
        """
        Predict signal quality using a more reliable approach until we have real data
        """
        try:
            # 1. Price-action based features
            rsi = market_features.get('rsi', 50)
            adx = market_features.get('adx', 25)
            regime = market_features.get('regime', 'neutral')
            
            # 2. Base score calculation
            quality_score = 5.0  # Neutral starting point
            
            # 3. Adjust for extreme RSI (strong signals)
            if rsi < 20 or rsi > 80:
                quality_score += 1.5  # Strong overbought/oversold
            elif rsi < 30 or rsi > 70:
                quality_score += 1.0  # Moderate overbought/oversold
                
            # 4. Adjust for trend strength
            if adx > 30:
                quality_score += 1.0  # Strong trend
            elif adx < 15:
                quality_score -= 0.5  # Weak trend

            # 5. Adjust for market regime
            if regime == 'ranging' and (rsi < 40 or rsi > 60):
                quality_score += 1.0  # Mean reversion in ranging market
            elif regime == 'strong_uptrend' and rsi > 50:
                quality_score += 1.0  # Trend continuation in uptrend
            elif regime == 'strong_downtrend' and rsi < 50:
                quality_score += 1.0  # Trend continuation in downtrend
                
            # 6. Limit to reasonable range
            quality_score = min(max(quality_score, 3.0), 8.5)
            
            logger.info(f"Calculated signal quality: {quality_score:.2f}/10")
            return quality_score
        
        except Exception as e:
            logger.error(f"Error in signal quality prediction: {e}")
            return 5.0  # Default neutral quality

    #def _predict_signal_quality(self, market_features):  ############ QUALITY PREDICTOR NOT WORKING AS INTENDED ###############
        """
        Predict the quality of a potential trading signal using the quality predictor model
        """
        try:
            # First make sure the model is initialized
            if "quality_predictor" not in self.models or self.models["quality_predictor"] is None:
                self._initialize_quality_predictor()
                
            # If model still not available, use a simple heuristic
            if "quality_predictor" not in self.models or self.models["quality_predictor"] is None:
                # Simple quality score based on RSI and ADX
                rsi = market_features.get('rsi', 50)
                adx = market_features.get('adx', 25)
                
                if rsi < 30 or rsi > 70:  # Overbought/oversold
                    quality = 0.7
                elif adx > 25:  # Strong trend
                    quality = 0.8
                else:
                    quality = 0.5
                    
                return quality * 10  # Scale to 0-10 range
            
            # Extract features in the correct order
            feature_list = ['rsi', 'adx', 'atr_percent', 'volume_ratio', 'macd']
            X = [market_features.get(f, 0) for f in feature_list]
            
            # Make prediction
            X = np.array(X).reshape(1, -1)  # Reshape for single sample prediction
            predicted_quality = self.models["quality_predictor"].predict(X)[0]
            
            # Convert to quality score (0-10 scale)
            # Map from profit prediction to quality score
            if predicted_quality > 0:
                # Positive profit prediction - higher quality
                quality_score = min(10, predicted_quality * 5 + 5)
            else:
                # Negative profit prediction - lower quality
                quality_score = max(0, 5 + predicted_quality * 5)
                
            return quality_score
            
        except Exception as e:
            logger.error(f"Error predicting signal quality: {e}")
            return 5.0  # Default neutral quality

    ##################################################################################################################################################
    ################################################################ Orderbook Analysis ##############################################################
    ##################################################################################################################################################

    def _analyze_orderbook_for_signals(self):
        """
        Analyzes orderbook data to extract trading signals with their strength
        
        Returns:
        --------
        tuple: (overall_score, list_of_signals)
            - overall_score: float between 0-10 indicating strength of orderbook signals
            - list_of_signals: list of dict containing signal details
        """
        orderbook_score = 0
        signals = []
        
        try:
            if not self.orderbook_data:
                logger.warning("No orderbook data available for analysis")
                return 0, []
            
            # 1. Extract liquidity walls and barriers
            try:
                if 'liquidity' in self.orderbook_data:
                    liquidity_data = self.orderbook_data['liquidity']
                    
                    # Process bid walls (support levels)
                    if 'bid_walls' in liquidity_data:
                        bid_walls = liquidity_data['bid_walls']
                        for wall in bid_walls:
                            if isinstance(wall, list) and len(wall) >= 3:
                                price, volume, impact = wall[:3]
                                if impact > 5:  # Only significant walls
                                    signals.append({
                                        'type': 'bid_wall',
                                        'price': float(price),
                                        'strength': float(impact),
                                        'description': f"Strong buy wall at {price}"
                                    })
                                    logger.info(f"Support detected: Buy wall at {price} (strength: {impact})")
                    
                    # Process ask walls (resistance levels)
                    if 'ask_walls' in liquidity_data:
                        ask_walls = liquidity_data['ask_walls']
                        for wall in ask_walls:
                            if isinstance(wall, list) and len(wall) >= 3:
                                price, volume, impact = wall[:3]
                                if impact > 5:  # Only significant walls
                                    signals.append({
                                        'type': 'ask_wall',
                                        'price': float(price),
                                        'strength': float(impact),
                                        'description': f"Strong sell wall at {price}"
                                    })
                                    logger.info(f"Resistance detected: Sell wall at {price} (strength: {impact})")
                    
                    # Process liquidity barriers
                    if 'barriers' in liquidity_data:
                        barriers_data = liquidity_data['barriers']
                        if isinstance(barriers_data, dict):
                            barriers = barriers_data.get('barriers', [])
                            for barrier in barriers:
                                barrier_type = barrier.get('type', '')
                                barrier_price = barrier.get('price', 0)
                                barrier_strength = barrier.get('strength', 0)
                                
                                if barrier_strength > 5:  # Only significant barriers
                                    signal_type = f"{barrier_type}_barrier"
                                    signals.append({
                                        'type': signal_type,
                                        'price': float(barrier_price),
                                        'strength': float(barrier_strength),
                                        'description': f"Strong {barrier_type} at {barrier_price}"
                                    })
                                    logger.info(f"Liquidity barrier: {barrier_type} at {barrier_price} (strength: {barrier_strength})")
            except Exception as e:
                logger.error(f"Error processing liquidity data: {e}")
            
            # 2. Extract imbalance zones
            try:
                if 'liquidity' in self.orderbook_data and 'imbalance' in self.orderbook_data['liquidity']:
                    imbalance_data = self.orderbook_data['liquidity']['imbalance']
                    if isinstance(imbalance_data, dict):
                        zones = imbalance_data.get('zones', [])
                        for zone in zones:
                            zone_type = zone.get('type', '')
                            zone_center = zone.get('center', 0)
                            zone_range = zone.get('price_range', [0, 0])
                            zone_strength = zone.get('strength', 0)
                            
                            if zone_strength > 5:  # Only significant imbalances
                                signal_type = f"{zone_type}_imbalance"
                                signals.append({
                                    'type': signal_type,
                                    'price': float(zone_center),
                                    'range': [float(zone_range[0]), float(zone_range[1])],
                                    'strength': float(zone_strength),
                                    'description': f"Order imbalance: {zone_type} at {zone_center}"
                                })
                                logger.info(f"Order imbalance: {zone_type} at {zone_center} (strength: {zone_strength})")
            except Exception as e:
                logger.error(f"Error processing imbalance data: {e}")
            
            # 3. Extract orderbook patterns
            try:
                if 'l3_analysis' in self.orderbook_data and 'patterns' in self.orderbook_data['l3_analysis']:
                    patterns_data = self.orderbook_data['l3_analysis']['patterns']
                    if isinstance(patterns_data, dict):
                        detected_patterns = patterns_data.get('detected_patterns', [])
                        for pattern in detected_patterns:
                            pattern_name = pattern.get('name', '')
                            pattern_conf = pattern.get('confidence', 5)
                            pattern_pred = pattern.get('prediction', '')
                            
                            if pattern_conf > 5:  # Only confident patterns
                                signals.append({
                                    'type': pattern_name,
                                    'strength': float(pattern_conf),
                                    'bias': pattern_pred,
                                    'description': pattern.get('description', f"{pattern_name} pattern")
                                })
                                logger.info(f"Orderbook pattern: {pattern_name} - {pattern_pred} (confidence: {pattern_conf})")
                        
                        # Check for composite signal
                        if 'composite_signal' in patterns_data:
                            comp_signal = patterns_data['composite_signal']
                            signal_bias = comp_signal.get('bias', 'neutral')
                            signal_strength = comp_signal.get('strength', 0)
                            
                            if signal_strength > 0.3:  # Only meaningful signals
                                signals.append({
                                    'type': f"composite_{signal_bias}",
                                    'strength': float(signal_strength * 10),  # Scale to 0-10
                                    'bias': signal_bias,
                                    'description': f"Overall orderbook bias: {signal_bias}"
                                })
                                logger.info(f"Composite orderbook bias: {signal_bias} (strength: {signal_strength:.2f})")
            except Exception as e:
                logger.error(f"Error processing pattern data: {e}")
            
            # 4. Check for anomalies and potential manipulation
            try:
                if 'l3_analysis' in self.orderbook_data and 'anomalies' in self.orderbook_data['l3_analysis']:
                    anomaly_data = self.orderbook_data['l3_analysis']['anomalies']
                    if isinstance(anomaly_data, dict):
                        # Check for persistent anomalies
                        persistent_anomalies = anomaly_data.get('persistent_anomalies', [])
                        for anomaly in persistent_anomalies:
                            anomaly_type = anomaly.get('type', '')
                            anomaly_severity = anomaly.get('severity', 0)
                            anomaly_price = anomaly.get('price', 0)
                            
                            if anomaly_severity > 6:  # Only severe anomalies
                                # Determine if bullish or bearish signal
                                if 'bid' in anomaly_type.lower():
                                    bias = 'bullish'
                                elif 'ask' in anomaly_type.lower():
                                    bias = 'bearish'
                                else:
                                    bias = 'neutral'
                                    
                                signals.append({
                                    'type': f"anomaly_{anomaly_type}",
                                    'price': float(anomaly_price) if anomaly_price else None,
                                    'strength': float(anomaly_severity),
                                    'bias': bias,
                                    'description': f"Persistent anomaly: {anomaly_type}"
                                })
                                logger.info(f"Orderbook anomaly: {anomaly_type} (severity: {anomaly_severity})")
                
                # Check for market manipulation
                if 'l3_analysis' in self.orderbook_data and 'manipulation' in self.orderbook_data['l3_analysis']:
                    manipulation_data = self.orderbook_data['l3_analysis']['manipulation']
                    if isinstance(manipulation_data, dict) and manipulation_data.get('detected', False):
                        manip_type = manipulation_data.get('type', '')
                        manip_confidence = manipulation_data.get('confidence', 0)
                        
                        if manip_confidence > 6:
                            logger.warning(f"Potential market manipulation detected: {manip_type}")
                            # If manipulation is detected, reduce the strength of all signals
                            for signal in signals:
                                signal['strength'] *= 0.7
                                signal['description'] += " (reduced due to potential manipulation)"
                            
                            # Add a warning signal
                            signals.append({
                                'type': f"manipulation_{manip_type}",
                                'strength': float(manip_confidence),
                                'bias': 'neutral',  # Manipulation doesn't have clear direction
                                'description': f"Potential market manipulation: {manip_type}"
                            })
            except Exception as e:
                logger.error(f"Error processing anomaly data: {e}")
            
            # 5. Process order flow analysis
            try:
                if 'l3_analysis' in self.orderbook_data and 'footprint' in self.orderbook_data['l3_analysis']:
                    footprint_data = self.orderbook_data['l3_analysis']['footprint']
                    if isinstance(footprint_data, dict):
                        # Check for strategic patterns
                        strategic_patterns = footprint_data.get('strategic_patterns', [])
                        for pattern in strategic_patterns:
                            pattern_type = pattern.get('type', '')
                            pattern_interp = pattern.get('interpretation', '')
                            
                            # Determine bias based on interpretation
                            bias = 'neutral'
                            if 'accumulation' in pattern_interp.lower():
                                bias = 'bullish'
                            elif 'distribution' in pattern_interp.lower():
                                bias = 'bearish'
                            
                            # Assign a default strength
                            strength = 7.0
                            
                            signals.append({
                                'type': f"order_flow_{pattern_type}",
                                'strength': strength,
                                'bias': bias,
                                'description': pattern_interp
                            })
                            logger.info(f"Order flow pattern: {pattern_type} - {pattern_interp}")
            except Exception as e:
                logger.error(f"Error processing order flow data: {e}")
            
            # 6. Calculate overall orderbook score
            # Sum the strength of bullish and bearish signals
            bullish_strength = 0
            bearish_strength = 0
            neutral_strength = 0
            
            for signal in signals:
                signal_strength = signal.get('strength', 0)
                signal_bias = signal.get('bias', '')
                
                if signal_bias == 'bullish' or 'bull' in signal_bias or 'support' in signal.get('type', ''):
                    bullish_strength += signal_strength
                elif signal_bias == 'bearish' or 'bear' in signal_bias or 'resistance' in signal.get('type', ''):
                    bearish_strength += signal_strength
                else:
                    neutral_strength += signal_strength
                    
            # Calculate overall score as difference between bullish and bearish signals
            if bullish_strength > bearish_strength:
                orderbook_score = bullish_strength - bearish_strength * 0.5
                logger.info(f"Overall orderbook bias: BULLISH (score: {orderbook_score:.2f})")
            elif bearish_strength > bullish_strength:
                orderbook_score = bearish_strength - bullish_strength * 0.5
                logger.info(f"Overall orderbook bias: BEARISH (score: {orderbook_score:.2f})")
            else:
                orderbook_score = neutral_strength * 0.3
                logger.info(f"Overall orderbook bias: NEUTRAL (score: {orderbook_score:.2f})")
                
            # Cap at 10
            orderbook_score = min(10.0, orderbook_score)
            
            return orderbook_score, signals
        
        except Exception as e:
            logger.error(f"Error in orderbook analysis: {e}")
            return 0, []

    def _generate_l2_heatmap(self, orderbook, bins=50):
        """Generates a detailed L2 liquidity heatmap to visualize order distribution"""
        try:
            # Extract prices and volumes from bids and asks
            bid_prices = [order[0] for order in orderbook["bids"]]
            bid_volumes = [order[1] for order in orderbook["bids"]]
            ask_prices = [order[0] for order in orderbook["asks"]]
            ask_volumes = [order[1] for order in orderbook["asks"]]

            if not bid_prices or not ask_prices:
                return {"status": "error", "reason": "orderbook_empty"}

            # Calculate price range for the entire heatmap
            min_price = min(min(bid_prices), min(ask_prices)) * 0.99
            max_price = max(max(bid_prices), max(ask_prices)) * 1.01

            # Create uniform price bins
            price_bins = np.linspace(min_price, max_price, bins + 1)
            bin_centers = [(price_bins[i] + price_bins[i + 1]) / 2 for i in range(bins)]

            # Initialize arrays for volume densities
            bid_density = np.zeros(bins)
            ask_density = np.zeros(bins)

            # Populate density arrays
            for price, volume in zip(bid_prices, bid_volumes):
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < bins:
                    bid_density[bin_idx] += volume

            for price, volume in zip(ask_prices, ask_volumes):
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < bins:
                    ask_density[bin_idx] += volume

            # Normalize to have values between 0 and 10
            max_density = max(max(bid_density), max(ask_density))
            if max_density > 0:
                bid_density = bid_density / max_density * 10
                ask_density = ask_density / max_density * 10

            # Identify liquidity gaps (zones with little depth)
            liquidity_gaps = []
            for i in range(1, bins - 1):
                if bid_density[i] < 0.2 * np.mean(bid_density) and ask_density[
                    i
                ] < 0.2 * np.mean(ask_density):
                    gap_strength = 10 - ((bid_density[i] + ask_density[i]) / 2)
                    if gap_strength > 7:  # Report only significant gaps
                        liquidity_gaps.append(
                            {
                                "price": bin_centers[i],
                                "strength": gap_strength,
                                "width": price_bins[i + 1] - price_bins[i],
                            }
                        )

            # Build the heatmap with analysis
            heatmap = {
                "price_bins": [float(p) for p in price_bins],
                "bin_centers": [float(p) for p in bin_centers],
                "bid_density": [float(d) for d in bid_density],
                "ask_density": [float(d) for d in ask_density],
                "mid_price": float(
                    (orderbook["bids"][0][0] + orderbook["asks"][0][0]) / 2
                ),
                "liquidity_gaps": liquidity_gaps,
                "analysis": {"top_bid_clusters": [], "top_ask_clusters": []},
            }

            # Identify the 3 most significant liquidity clusters
            for idx in np.argsort(bid_density)[-3:]:
                if bid_density[idx] > 0:
                    heatmap["analysis"]["top_bid_clusters"].append(
                        {
                            "price": float(bin_centers[idx]),
                            "relative_strength": float(bid_density[idx] / 10),
                            "bin_index": int(idx),
                        }
                    )

            for idx in np.argsort(ask_density)[-3:]:
                if ask_density[idx] > 0:
                    heatmap["analysis"]["top_ask_clusters"].append(
                        {
                            "price": float(bin_centers[idx]),
                            "relative_strength": float(ask_density[idx] / 10),
                            "bin_index": int(idx),
                        }
                    )

            # Sort by price (ascending bids, descending asks)
            heatmap["analysis"]["top_bid_clusters"].sort(
                key=lambda x: x["price"], reverse=True
            )
            heatmap["analysis"]["top_ask_clusters"].sort(key=lambda x: x["price"])

            # NEW: Compute gradient of density to identify support/resistance areas
            bid_gradient = np.gradient(bid_density)
            ask_gradient = np.gradient(ask_density)

            # Identify areas of high gradient (rapid change in liquidity)
            support_levels = []
            resistance_levels = []

            # Supports (positive gradient in bids)
            for i in range(1, bins - 1):
                if (
                    bid_gradient[i] > 0 and bid_gradient[i - 1] < 0
                ):  # Sign change (minimum)
                    if bid_density[i] > np.mean(bid_density):
                        support_levels.append(
                            {
                                "price": float(bin_centers[i]),
                                "strength": float(
                                    bid_density[i] / 10 * 5 + abs(bid_gradient[i]) * 5
                                ),
                                "gradient": float(bid_gradient[i]),
                            }
                        )

            # Resistances (positive gradient in asks)
            for i in range(1, bins - 1):
                if (
                    ask_gradient[i] > 0 and ask_gradient[i - 1] < 0
                ):  # Sign change (minimum)
                    if ask_density[i] > np.mean(ask_density):
                        resistance_levels.append(
                            {
                                "price": float(bin_centers[i]),
                                "strength": float(
                                    ask_density[i] / 10 * 5 + abs(ask_gradient[i]) * 5
                                ),
                                "gradient": float(ask_gradient[i]),
                            }
                        )

            # Sort by strength
            support_levels.sort(key=lambda x: x["strength"], reverse=True)
            resistance_levels.sort(key=lambda x: x["strength"], reverse=True)

            # Add to heatmap
            heatmap["analysis"]["support_levels"] = support_levels[:3]  # Top 3 supports
            heatmap["analysis"]["resistance_levels"] = resistance_levels[
                :3
            ]  # Top 3 resistances

            return heatmap

        except Exception as e:
            logger.error(f"Error in generating L2 heatmap: {e}")
            return {"status": "error", "reason": str(e)}

    def _detect_orderbook_patterns(self, orderbook):
        """Detects known orderbook patterns that have predictive value"""
        try:
            if not orderbook or not orderbook["bids"] or not orderbook["asks"]:
                return {"detected_patterns": []}

            # Extract prices and volumes
            bid_prices = np.array([order[0] for order in orderbook["bids"]])
            bid_volumes = np.array([order[1] for order in orderbook["bids"]])
            ask_prices = np.array([order[0] for order in orderbook["asks"]])
            ask_volumes = np.array([order[1] for order in orderbook["asks"]])

            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2

            patterns = []

            # Pattern 1: "Buy wall" - Strong support with large orders
            if len(bid_volumes) > 5:
                avg_top_5_bid_vol = np.mean(bid_volumes[:5])
                if bid_volumes[0] > avg_top_5_bid_vol * 3:
                    patterns.append(
                        {
                            "name": "buy_wall",
                            "price": float(best_bid),
                            "confidence": min(
                                10, float(bid_volumes[0] / avg_top_5_bid_vol)
                            ),
                            "prediction": "bullish",
                            "description": "Strong buying interest at current level",
                        }
                    )

            # Pattern 2: "Sell wall" - Strong resistance with large orders
            if len(ask_volumes) > 5:
                avg_top_5_ask_vol = np.mean(ask_volumes[:5])
                if ask_volumes[0] > avg_top_5_ask_vol * 3:
                    patterns.append(
                        {
                            "name": "sell_wall",
                            "price": float(best_ask),
                            "confidence": min(
                                10, float(ask_volumes[0] / avg_top_5_ask_vol)
                            ),
                            "prediction": "bearish",
                            "description": "Strong selling interest at current level",
                        }
                    )

            # Pattern 3: "Iceberg" - Small orders at the top, large orders behind
            if len(bid_volumes) > 10 and bid_volumes[0] < np.mean(bid_volumes[1:10]):
                patterns.append(
                    {
                        "name": "buy_iceberg",
                        "price": float(best_bid),
                        "confidence": min(
                            10, float(np.mean(bid_volumes[1:10]) / bid_volumes[0])
                        ),
                        "prediction": "bullish",
                        "description": "Hidden buying interest behind top order",
                    }
                )

            if len(ask_volumes) > 10 and ask_volumes[0] < np.mean(ask_volumes[1:10]):
                patterns.append(
                    {
                        "name": "sell_iceberg",
                        "price": float(best_ask),
                        "confidence": min(
                            10, float(np.mean(ask_volumes[1:10]) / ask_volumes[0])
                        ),
                        "prediction": "bearish",
                        "description": "Hidden selling interest behind top order",
                    }
                )

            # Pattern 4: "Thin Zone" - Significant liquidity gap
            if len(bid_prices) > 10 and len(ask_prices) > 10:
                bid_gaps = bid_prices[:-1] - bid_prices[1:]
                ask_gaps = ask_prices[1:] - ask_prices[:-1]

                max_bid_gap_idx = np.argmax(bid_gaps)
                max_ask_gap_idx = np.argmax(ask_gaps)

                if bid_gaps[max_bid_gap_idx] > np.mean(bid_gaps) * 5:
                    patterns.append(
                        {
                            "name": "thin_support",
                            "price_range": [
                                float(bid_prices[max_bid_gap_idx + 1]),
                                float(bid_prices[max_bid_gap_idx]),
                            ],
                            "confidence": min(
                                10, float(bid_gaps[max_bid_gap_idx] / np.mean(bid_gaps))
                            ),
                            "prediction": "bearish_breakout_potential",
                            "description": "Thin liquidity zone below current price",
                        }
                    )

                if ask_gaps[max_ask_gap_idx] > np.mean(ask_gaps) * 5:
                    patterns.append(
                        {
                            "name": "thin_resistance",
                            "price_range": [
                                float(ask_prices[max_ask_gap_idx]),
                                float(ask_prices[max_ask_gap_idx + 1]),
                            ],
                            "confidence": min(
                                10, float(ask_gaps[max_ask_gap_idx] / np.mean(ask_gaps))
                            ),
                            "prediction": "bullish_breakout_potential",
                            "description": "Thin liquidity zone above current price",
                        }
                    )

            # Pattern 5: "Liquidity Imbalance" - Significantly more orders on one side
            total_bid_volume = np.sum(bid_volumes)
            total_ask_volume = np.sum(ask_volumes)

            if total_bid_volume > total_ask_volume * 2:
                patterns.append(
                    {
                        "name": "buyer_dominance",
                        "ratio": float(total_bid_volume / total_ask_volume),
                        "confidence": min(
                            10, float(total_bid_volume / total_ask_volume / 2)
                        ),
                        "prediction": "bullish",
                        "description": "Buyers significantly outweighing sellers",
                    }
                )
            elif total_ask_volume > total_bid_volume * 2:
                patterns.append(
                    {
                        "name": "seller_dominance",
                        "ratio": float(total_ask_volume / total_bid_volume),
                        "confidence": min(
                            10, float(total_ask_volume / total_bid_volume / 2)
                        ),
                        "prediction": "bearish",
                        "description": "Sellers significantly outweighing buyers",
                    }
                )

            # Pattern 6: "Spoofing Potential" - Unusually large orders far from mid price
            far_bid_volumes = (
                bid_volumes[bid_prices < mid_price * 0.98]
                if len(bid_prices[bid_prices < mid_price * 0.98]) > 0
                else []
            )
            far_ask_volumes = (
                ask_volumes[ask_prices > mid_price * 1.02]
                if len(ask_prices[ask_prices > mid_price * 1.02]) > 0
                else []
            )

            if len(far_bid_volumes) > 0 and len(bid_volumes) > 10:
                max_far_bid_vol = np.max(far_bid_volumes)
                avg_bid_vol = np.mean(bid_volumes)

                if max_far_bid_vol > avg_bid_vol * 5:
                    patterns.append(
                        {
                            "name": "potential_buy_spoofing",
                            "ratio": float(max_far_bid_vol / avg_bid_vol),
                            "confidence": min(
                                10, float(max_far_bid_vol / avg_bid_vol / 5)
                            ),
                            "prediction": "false_support",
                            "description": "Potential fake buying pressure",
                        }
                    )

            if len(far_ask_volumes) > 0 and len(ask_volumes) > 10:
                max_far_ask_vol = np.max(far_ask_volumes)
                avg_ask_vol = np.mean(ask_volumes)

                if max_far_ask_vol > avg_ask_vol * 5:
                    patterns.append(
                        {
                            "name": "potential_sell_spoofing",
                            "ratio": float(max_far_ask_vol / avg_ask_vol),
                            "confidence": min(
                                10, float(max_far_ask_vol / avg_ask_vol / 5)
                            ),
                            "prediction": "false_resistance",
                            "description": "Potential fake selling pressure",
                        }
                    )

            # NEW: Pattern 7: "Volume Cliff" - Sudden drop in liquidity
            if len(bid_volumes) > 5:
                bid_vol_drops = bid_volumes[:-1] / (
                    bid_volumes[1:] + 0.000001
                )  # Avoid division by zero
                max_drop_idx = np.argmax(bid_vol_drops)

                if bid_vol_drops[max_drop_idx] > 5:  # Volume drops by 5x or more
                    patterns.append(
                        {
                            "name": "bid_volume_cliff",
                            "price": float(bid_prices[max_drop_idx + 1]),
                            "ratio": float(bid_vol_drops[max_drop_idx]),
                            "confidence": min(
                                10, float(bid_vol_drops[max_drop_idx] / 5)
                            ),
                            "prediction": "support_breakdown_risk",
                            "description": "Sharp drop in buyer support below this level",
                        }
                    )

            if len(ask_volumes) > 5:
                ask_vol_drops = ask_volumes[:-1] / (
                    ask_volumes[1:] + 0.000001
                )  # Avoid division by zero
                max_drop_idx = np.argmax(ask_vol_drops)

                if ask_vol_drops[max_drop_idx] > 5:  # Volume drops by 5x or more
                    patterns.append(
                        {
                            "name": "ask_volume_cliff",
                            "price": float(ask_prices[max_drop_idx + 1]),
                            "ratio": float(ask_vol_drops[max_drop_idx]),
                            "confidence": min(
                                10, float(ask_vol_drops[max_drop_idx] / 5)
                            ),
                            "prediction": "resistance_breakthrough_potential",
                            "description": "Sharp drop in seller resistance above this level",
                        }
                    )

            # NEW: Pattern 8: "Order Block" - Detection of institutional order blocks
            # Look for dense clusters of orders at specific price levels
            try:
                # Apply density-based clustering to find order blocks
                if self.params["order_block_detection"] and len(bid_prices) > 10:
                    X_bid = bid_prices.reshape(-1, 1)
                    clustering_bid = DBSCAN(eps=0.0005, min_samples=3).fit(X_bid)

                    # Check for clusters
                    if hasattr(clustering_bid, "labels_"):
                        labels_bid = clustering_bid.labels_
                        unique_labels_bid = set(labels_bid)

                        for label in unique_labels_bid:
                            if label != -1:  # Skip noise
                                mask = labels_bid == label
                                if np.sum(mask) >= 3:  # At least 3 orders in cluster
                                    cluster_prices = bid_prices[mask]
                                    cluster_volumes = bid_volumes[mask]

                                    avg_price = np.mean(cluster_prices)
                                    total_volume = np.sum(cluster_volumes)

                                    # Check if this is a significant order block
                                    if total_volume > np.mean(bid_volumes) * 3:
                                        patterns.append(
                                            {
                                                "name": "bid_order_block",
                                                "price": float(avg_price),
                                                "volume": float(total_volume),
                                                "order_count": int(np.sum(mask)),
                                                "confidence": min(
                                                    10,
                                                    float(
                                                        total_volume
                                                        / np.mean(bid_volumes)
                                                        / 2
                                                    ),
                                                ),
                                                "prediction": "strong_support",
                                                "description": "Dense block of buy orders indicating institutional interest",
                                            }
                                        )

                if self.params["order_block_detection"] and len(ask_prices) > 10:
                    X_ask = ask_prices.reshape(-1, 1)
                    clustering_ask = DBSCAN(eps=0.0005, min_samples=3).fit(X_ask)

                    # Check for clusters
                    if hasattr(clustering_ask, "labels_"):
                        labels_ask = clustering_ask.labels_
                        unique_labels_ask = set(labels_ask)

                        for label in unique_labels_ask:
                            if label != -1:  # Skip noise
                                mask = labels_ask == label
                                if np.sum(mask) >= 3:  # At least 3 orders in cluster
                                    cluster_prices = ask_prices[mask]
                                    cluster_volumes = ask_volumes[mask]

                                    avg_price = np.mean(cluster_prices)
                                    total_volume = np.sum(cluster_volumes)

                                    # Check if this is a significant order block
                                    if total_volume > np.mean(ask_volumes) * 3:
                                        patterns.append(
                                            {
                                                "name": "ask_order_block",
                                                "price": float(avg_price),
                                                "volume": float(total_volume),
                                                "order_count": int(np.sum(mask)),
                                                "confidence": min(
                                                    10,
                                                    float(
                                                        total_volume
                                                        / np.mean(ask_volumes)
                                                        / 2
                                                    ),
                                                ),
                                                "prediction": "strong_resistance",
                                                "description": "Dense block of sell orders indicating institutional interest",
                                            }
                                        )
            except Exception as e:
                logger.warning(f"Error in order block detection: {e}")

            # Sort patterns by confidence
            patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)

            # NEW: Calculate a composite orderbook signal
            bullish_signals = [
                p
                for p in patterns
                if p.get("prediction", "").startswith("bull")
                or p.get("name", "").startswith("buy")
            ]
            bearish_signals = [
                p
                for p in patterns
                if p.get("prediction", "").startswith("bear")
                or p.get("name", "").startswith("sell")
            ]

            bullish_score = (
                sum(p.get("confidence", 0) for p in bullish_signals) / 10
                if bullish_signals
                else 0
            )
            bearish_score = (
                sum(p.get("confidence", 0) for p in bearish_signals) / 10
                if bearish_signals
                else 0
            )

            composite_signal = {
                "bullish_score": float(bullish_score),
                "bearish_score": float(bearish_score),
                "net_score": float(bullish_score - bearish_score),
                "bias": (
                    "bullish"
                    if bullish_score > bearish_score
                    else "bearish" if bearish_score > bullish_score else "neutral"
                ),
                "strength": float(abs(bullish_score - bearish_score)),
            }

            return {
                "detected_patterns": patterns,
                "count": len(patterns),
                "composite_signal": composite_signal,
            }

        except Exception as e:
            logger.error(f"Error in detecting orderbook patterns: {e}")
            return {"detected_patterns": []}

    def _detect_orderbook_anomalies(self, orderbook):
        """Uses unsupervised learning to detect unusual orderbook structures"""
        try:
            if not orderbook or not orderbook["bids"] or not orderbook["asks"]:
                return {"anomalies": []}

            # Extract prices and volumes
            bid_prices = np.array([order[0] for order in orderbook["bids"]])
            bid_volumes = np.array([order[1] for order in orderbook["bids"]])
            ask_prices = np.array([order[0] for order in orderbook["asks"]])
            ask_volumes = np.array([order[1] for order in orderbook["asks"]])

            # Safety check
            if len(bid_prices) < 10 or len(ask_prices) < 10:
                return {"anomalies": []}

            # Create feature vectors
            bid_features = []
            ask_features = []

            # For top 10 bids and asks
            for i in range(min(10, len(bid_prices))):
                bid_features.append(
                    [bid_volumes[i], 0 if i == 0 else bid_prices[i - 1] - bid_prices[i]]
                )

            for i in range(min(10, len(ask_prices))):
                ask_features.append(
                    [ask_volumes[i], 0 if i == 0 else ask_prices[i] - ask_prices[i - 1]]
                )

            # Combine features
            features = np.array(bid_features + ask_features)

            # Skip if features are insufficient
            if len(features) < 5:
                return {"anomalies": []}

            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)

            # Initialize or use existing anomaly detector
            try:
                if self.models["anomaly_detector"] is None:
                    # Use IsolationForest for anomaly detection
                    self.models["anomaly_detector"] = IsolationForest(
                        contamination=0.1,  # Expect about 10% of data to be anomalous
                        random_state=42,
                    )
                
                # Assicurati che il modello sia sempre addestrato con i dati attuali
                # In produzione, potresti voler conservare un training set più grande
                # e riaddestrate solo periodicamente
                self.models["anomaly_detector"].fit(normalized_features)
                
                # Get anomaly scores (lower = more anomalous)
                scores = self.models["anomaly_detector"].decision_function(
                    normalized_features
                )
            except Exception as e:
                logger.error(f"Error nel training del modello di anomalia: {e}")
                return {"anomalies": []}

            # Identify anomalies (scores close to -1 are anomalies)
            anomalies = []
            for i, score in enumerate(scores):
                if score < -0.5:  # Strong anomaly threshold
                    if i < len(bid_features):
                        # Bid-side anomaly
                        idx = i
                        anomalies.append(
                            {
                                "type": "bid",
                                "index": idx,
                                "price": float(bid_prices[idx]),
                                "volume": float(bid_volumes[idx]),
                                "score": float(score),
                                "severity": min(10, float((1 + score) * -10)),
                            }
                        )
                    else:
                        # Ask-side anomaly
                        idx = i - len(bid_features)
                        anomalies.append(
                            {
                                "type": "ask",
                                "index": idx,
                                "price": float(ask_prices[idx]),
                                "volume": float(ask_volumes[idx]),
                                "score": float(score),
                                "severity": min(10, float((1 + score) * -10)),
                            }
                        )

            # NEW: Advanced anomaly analysis
            # Look for patterns in anomalies
            anomaly_patterns = []

            # Check for clusters of anomalies (multiple anomalies near each other)
            bid_anomalies = [a for a in anomalies if a["type"] == "bid"]
            ask_anomalies = [a for a in anomalies if a["type"] == "ask"]

            # Group bid anomalies by price proximity
            if len(bid_anomalies) >= 2:
                bid_anomaly_prices = np.array([a["price"] for a in bid_anomalies])

                # Check if anomalies are clustered
                price_diffs = np.abs(bid_anomaly_prices[:-1] - bid_anomaly_prices[1:])
                if np.any(price_diffs < 0.005):  # Anomalies within 0.5% of each other
                    anomaly_patterns.append(
                        {
                            "type": "clustered_bid_anomalies",
                            "count": len(bid_anomalies),
                            "price_range": [
                                float(min(bid_anomaly_prices)),
                                float(max(bid_anomaly_prices)),
                            ],
                            "interpretation": "Multiple unusual orders on buy side - potential accumulation",
                            "severity": min(10, len(bid_anomalies) * 2),
                        }
                    )

            # Group ask anomalies by price proximity
            if len(ask_anomalies) >= 2:
                ask_anomaly_prices = np.array([a["price"] for a in ask_anomalies])

                # Check if anomalies are clustered
                price_diffs = np.abs(ask_anomaly_prices[:-1] - ask_anomaly_prices[1:])
                if np.any(price_diffs < 0.005):  # Anomalies within 0.5% of each other
                    anomaly_patterns.append(
                        {
                            "type": "clustered_ask_anomalies",
                            "count": len(ask_anomalies),
                            "price_range": [
                                float(min(ask_anomaly_prices)),
                                float(max(ask_anomaly_prices)),
                            ],
                            "interpretation": "Multiple unusual orders on sell side - potential distribution",
                            "severity": min(10, len(ask_anomalies) * 2),
                        }
                    )

            # Sort anomalies by severity
            anomalies.sort(key=lambda x: x["severity"], reverse=True)

            # NEW: Check for persistent anomalies
            persistent_anomalies = []

            # If we have historical orderbooks, check if any anomalies persist
            if len(self.orderbook_history) > 1:
                prev_orderbook = self.orderbook_history[-1]
                if (
                    "l3_analysis" in prev_orderbook
                    and "anomalies" in prev_orderbook["l3_analysis"]
                ):
                    prev_anomalies = prev_orderbook["l3_analysis"]["anomalies"].get(
                        "anomalies", []
                    )

                    for anomaly in anomalies:
                        for prev_anomaly in prev_anomalies:
                            # If an anomaly at same price (within 0.1%) and same type persists
                            if (
                                prev_anomaly.get("type") == anomaly["type"]
                                and abs(prev_anomaly.get("price", 0) - anomaly["price"])
                                / anomaly["price"]
                                < 0.001
                            ):

                                persistent_anomalies.append(
                                    {
                                        "type": f"persistent_{anomaly['type']}_anomaly",
                                        "price": anomaly["price"],
                                        "severity": min(
                                            10, anomaly["severity"] * 1.3
                                        ),  # Boost severity for persistence
                                        "interpretation": f"Unusual {anomaly['type']} order persisting over time",
                                    }
                                )
                                break

            return {
                "anomalies": anomalies,
                "count": len(anomalies),
                "patterns": anomaly_patterns,
                "persistent_anomalies": persistent_anomalies,
            }

        except Exception as e:
            logger.error(f"Error in detecting orderbook anomalies: {e}")
            return {"anomalies": []}

    def _analyze_volume_distribution(self, orders, bins=20):
        """Analyzes the distribution of volumes in the orderbook"""
        try:
            if not orders or len(orders) < 5:
                return {"status": "insufficient_data"}

            # Extract prices and volumes
            prices = np.array([order[0] for order in orders])
            volumes = np.array([order[1] for order in orders])

            # Calculate basic statistics
            total_volume = np.sum(volumes)
            mean_volume = np.mean(volumes)
            median_volume = np.median(volumes)

            # Calculate volume concentration (approximated Gini coefficient)
            sorted_volumes = np.sort(volumes)
            cum_volumes = np.cumsum(sorted_volumes)
            cum_proportion = cum_volumes / cum_volumes[-1]
            gini = 1 - 2 * np.trapz(cum_proportion) / len(volumes)

            # Divide into bins and calculate concentration per bin
            price_min, price_max = np.min(prices), np.max(prices)
            price_bins = np.linspace(price_min, price_max, bins + 1)
            bin_centers = [(price_bins[i] + price_bins[i + 1]) / 2 for i in range(bins)]

            # Initialize array for volumes per bin
            bin_volumes = np.zeros(bins)

            # Populate bins with volumes
            for price, volume in zip(prices, volumes):
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < bins:
                    bin_volumes[bin_idx] += volume

            # Normalize bin volumes
            volume_distribution = (
                bin_volumes / total_volume if total_volume > 0 else bin_volumes
            )

            # Identify bins with anomalous concentration (>25% of total volume)
            concentration_anomalies = []
            for i, (center, volume) in enumerate(zip(bin_centers, volume_distribution)):
                if volume > 0.25:  # More than 25% of volume in a single bin
                    concentration_anomalies.append(
                        {
                            "price_center": float(center),
                            "volume_percentage": float(volume * 100),
                            "bin_index": i,
                        }
                    )

            # Calculate bin-to-bin volume ratio changes
            volume_ratio_changes = []
            for i in range(1, bins):
                if bin_volumes[i - 1] > 0:
                    ratio = bin_volumes[i] / bin_volumes[i - 1]
                    # Only include significant changes
                    if ratio > 3 or ratio < 0.33:
                        volume_ratio_changes.append(
                            {
                                "from_bin": i - 1,
                                "to_bin": i,
                                "from_price": float(bin_centers[i - 1]),
                                "to_price": float(bin_centers[i]),
                                "ratio": float(ratio),
                                "is_increase": ratio > 1,
                            }
                        )

            # NEW: Find peaks in volume distribution
            peaks, _ = find_peaks(
                bin_volumes, height=np.mean(bin_volumes) * 1.5, distance=2
            )
            volume_peaks = []

            for peak in peaks:
                volume_peaks.append(
                    {
                        "price": float(bin_centers[peak]),
                        "relative_volume": float(
                            bin_volumes[peak] / np.mean(bin_volumes)
                        ),
                        "bin_index": int(peak),
                    }
                )

            # Sort peaks by volume
            volume_peaks.sort(key=lambda x: x["relative_volume"], reverse=True)

            # Analysis result
            result = {
                "total_volume": float(total_volume),
                "mean_volume": float(mean_volume),
                "median_volume": float(median_volume),
                "gini_coefficient": float(gini),
                "bin_centers": [float(c) for c in bin_centers],
                "volume_distribution": [float(v) for v in volume_distribution],
                "distribution_skew": (
                    float(
                        np.sum((volumes - mean_volume) ** 3)
                        / (len(volumes) * np.std(volumes) ** 3)
                    )
                    if len(volumes) > 1 and np.std(volumes) > 0
                    else 0
                ),
                "concentration_anomalies": concentration_anomalies,
                "volume_ratio_changes": volume_ratio_changes,
                "distribution_uniformity": 1.0
                - float(gini),  # 1 = uniform distribution, 0 = extremely unbalanced
                "volume_peaks": volume_peaks,
            }

            return result

        except Exception as e:
            logger.error(f"Error in analyzing volume distribution: {e}")
            return {"status": "error", "reason": str(e)}

    def _identify_liquidity_barriers(self, orderbook):
        """Identifies significant liquidity barriers in the orderbook"""
        try:
            barriers = []

            # Check that orderbook_data is not None and has the expected structure
            if self.orderbook_data is None or "liquidity" not in self.orderbook_data:
                return {"barriers": [], "significant_barriers": False}

            # Extract data from liquidity walls and clusters
            bid_walls = self.orderbook_data["liquidity"].get("bid_walls", [])
            ask_walls = self.orderbook_data["liquidity"].get("ask_walls", [])

            # Process bid walls
            for wall in bid_walls:
                if isinstance(wall, list) and len(wall) >= 3:
                    price, volume, impact = wall[:3]
                    if impact > 5:  # Consider only significant walls
                        barriers.append(
                            {
                                "price": float(price),
                                "volume": float(volume),
                                "type": "support",
                                "source": "bid_wall",
                                "strength": float(impact),
                            }
                        )

            # Process ask walls
            for wall in ask_walls:
                if isinstance(wall, list) and len(wall) >= 3:
                    price, volume, impact = wall[:3]
                    if impact > 5:  # Consider only significant walls
                        barriers.append(
                            {
                                "price": float(price),
                                "volume": float(volume),
                                "type": "resistance",
                                "source": "ask_wall",
                                "strength": float(impact),
                            }
                        )

            # Use data from the heatmap if we haven't found barriers
            if not barriers and self.l2_heatmap and "analysis" in self.l2_heatmap:
                for cluster in self.l2_heatmap["analysis"].get("top_bid_clusters", []):
                    barriers.append(
                        {
                            "price": float(cluster["price"]),
                            "volume": float(cluster["relative_strength"] * 100),
                            "type": "support",
                            "source": "heatmap",
                            "strength": float(cluster["relative_strength"] * 10),
                        }
                    )

                for cluster in self.l2_heatmap["analysis"].get("top_ask_clusters", []):
                    barriers.append(
                        {
                            "price": float(cluster["price"]),
                            "volume": float(cluster["relative_strength"] * 100),
                            "type": "resistance",
                            "source": "heatmap",
                            "strength": float(cluster["relative_strength"] * 10),
                        }
                    )

            # Also look for liquidity gaps as potential barriers
            if self.l2_heatmap and "liquidity_gaps" in self.l2_heatmap:
                for gap in self.l2_heatmap["liquidity_gaps"]:
                    # A liquidity gap is a zone with little depth
                    # that can act as both support and resistance
                    barriers.append(
                        {
                            "price": float(gap["price"]),
                            "type": "gap",
                            "source": "liquidity_gap",
                            "strength": float(gap["strength"]),
                            "width": float(gap["width"]),
                        }
                    )

            # Sort by strength
            barriers.sort(key=lambda x: x["strength"], reverse=True)

            # Group barriers that are very close (within 0.2%)
            grouped_barriers = []
            if barriers:
                current_group = [barriers[0]]

                for i in range(1, len(barriers)):
                    barrier = barriers[i]
                    last_barrier = current_group[-1]

                    # Calculate percentage distance
                    price_distance = (
                        abs(barrier["price"] - last_barrier["price"])
                        / last_barrier["price"]
                    )

                    if price_distance < 0.002:  # Within 0.2%
                        # Add to current group
                        current_group.append(barrier)
                    else:
                        # Average of barriers in the current group
                        avg_price = sum(b["price"] for b in current_group) / len(
                            current_group
                        )
                        avg_strength = sum(b["strength"] for b in current_group) / len(
                            current_group
                        )
                        # Take the most common type in the group
                        types = [b["type"] for b in current_group]
                        most_common_type = max(set(types), key=types.count)

                        grouped_barriers.append(
                            {
                                "price": avg_price,
                                "type": most_common_type,
                                "strength": avg_strength,
                                "barrier_count": len(current_group),
                            }
                        )

                        # Start a new group
                        current_group = [barrier]

                # Add the last group
                if current_group:
                    avg_price = sum(b["price"] for b in current_group) / len(
                        current_group
                    )
                    avg_strength = sum(b["strength"] for b in current_group) / len(
                        current_group
                    )
                    types = [b["type"] for b in current_group]
                    most_common_type = max(set(types), key=types.count)

                    grouped_barriers.append(
                        {
                            "price": avg_price,
                            "type": most_common_type,
                            "strength": avg_strength,
                            "barrier_count": len(current_group),
                        }
                    )

            # Sort grouped barriers by strength
            grouped_barriers.sort(key=lambda x: x["strength"], reverse=True)

            # Analyze historical strength of these barriers
            if len(self.orderbook_history) > 5:
                for barrier in grouped_barriers:
                    # Track persistence of this barrier
                    persistence_count = 0
                    avg_historical_strength = 0

                    for i, hist_ob in enumerate(
                        reversed(list(self.orderbook_history)[:-1])
                    ):
                        if i >= 5:  # Check last 5 orderbooks
                            break

                        if (
                            "liquidity" in hist_ob
                            and "barriers" in hist_ob["liquidity"]
                        ):
                            hist_barriers = hist_ob["liquidity"]["barriers"].get(
                                "barriers", []
                            )

                            # Look for this barrier in history
                            for hist_barrier in hist_barriers:
                                if (
                                    abs(hist_barrier.get("price", 0) - barrier["price"])
                                    / barrier["price"]
                                    < 0.005
                                ):
                                    persistence_count += 1
                                    avg_historical_strength += hist_barrier.get(
                                        "strength", 0
                                    )
                                    break

                    # Calculate historical metrics
                    barrier["persistence"] = persistence_count / 5  # 0-1 scale
                    barrier["historical_strength"] = avg_historical_strength / max(
                        1, persistence_count
                    )

                    # Boost strength for persistent barriers
                    if barrier["persistence"] > 0.6:
                        barrier["strength"] = barrier["strength"] * (
                            1 + barrier["persistence"] * 0.2
                        )

            # Identify if there are particularly significant barriers
            significant_barriers = False
            most_significant_barrier = None
            if grouped_barriers and grouped_barriers[0]["strength"] > 7.5:
                significant_barriers = True
                most_significant_barrier = grouped_barriers[0]

            # NEW: Calculate confidence score for each barrier
            # This combines strength, persistence, and other factors
            for barrier in grouped_barriers:
                confidence_score = (
                    barrier["strength"] * 0.6
                )  # Base confidence on strength

                # Add persistence factor if available
                if "persistence" in barrier:
                    confidence_score += barrier["persistence"] * 10 * 0.4

                # Cap at 10
                barrier["confidence"] = min(10, confidence_score)

            # Filter barriers by confidence threshold
            if self.params["liquidity_confidence_threshold"] > 0:
                threshold = (
                    self.params["liquidity_confidence_threshold"] * 10
                )  # Scale to 0-10
                filtered_barriers = [
                    b for b in grouped_barriers if b.get("confidence", 0) >= threshold
                ]

                if filtered_barriers:
                    grouped_barriers = filtered_barriers

            return {
                "barriers": grouped_barriers,
                "significant_barriers": significant_barriers,
                "most_significant_barrier": most_significant_barrier,
            }

        except Exception as e:
            logger.error(f"Error in identifying liquidity barriers: {e}")
            return {"barriers": [], "significant_barriers": False}

    def _detect_imbalance_zones(self, orderbook, threshold=3.0):
        """Detects zones of significant imbalance between bid and ask"""
        try:
            # Extract prices and volumes
            bid_prices = np.array([order[0] for order in orderbook["bids"]])
            bid_volumes = np.array([order[1] for order in orderbook["bids"]])
            ask_prices = np.array([order[0] for order in orderbook["asks"]])
            ask_volumes = np.array([order[1] for order in orderbook["asks"]])

            if len(bid_prices) < 5 or len(ask_prices) < 5:
                return {"zones": []}

            # Define price bins for analysis
            min_price = min(np.min(bid_prices), np.min(ask_prices)) * 0.99
            max_price = max(np.max(bid_prices), np.max(ask_prices)) * 1.01

            # Create 20 uniform price bins
            bins = 20
            price_bins = np.linspace(min_price, max_price, bins + 1)
            bin_centers = [(price_bins[i] + price_bins[i + 1]) / 2 for i in range(bins)]

            # Calculate total volume for each bin
            bid_bin_volumes = np.zeros(bins)
            ask_bin_volumes = np.zeros(bins)

            for price, volume in zip(bid_prices, bid_volumes):
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < bins:
                    bid_bin_volumes[bin_idx] += volume

            for price, volume in zip(ask_prices, ask_volumes):
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < bins:
                    ask_bin_volumes[bin_idx] += volume

            # Calculate ratio between bid and ask for each bin (with zero division protection)
            imbalance_ratios = np.zeros(bins)
            for i in range(bins):
                if ask_bin_volumes[i] > 0 and bid_bin_volumes[i] > 0:
                    ratio = bid_bin_volumes[i] / ask_bin_volumes[i]
                    imbalance_ratios[i] = ratio
                elif bid_bin_volumes[i] > 0:
                    imbalance_ratios[i] = (
                        threshold + 1
                    )  # Strong bid imbalance (no asks)
                elif ask_bin_volumes[i] > 0:
                    imbalance_ratios[i] = 1 / (
                        threshold + 1
                    )  # Strong ask imbalance (no bids)
                else:
                    imbalance_ratios[i] = 1.0  # No volume on both sides

            # Identify zones of significant imbalance
            imbalance_zones = []
            for i in range(bins):
                ratio = imbalance_ratios[i]
                if ratio > threshold:  # More bids than asks (potential support)
                    imbalance_zones.append(
                        {
                            "price_range": [
                                float(price_bins[i]),
                                float(price_bins[i + 1]),
                            ],
                            "center": float(bin_centers[i]),
                            "type": "bid_dominance",
                            "ratio": float(ratio),
                            "strength": min(
                                10, float(ratio / threshold * 5)
                            ),  # Scale 0-10
                        }
                    )
                elif (
                    ratio < 1 / threshold
                ):  # More asks than bids (potential resistance)
                    inverse_ratio = 1 / ratio if ratio > 0 else threshold
                    imbalance_zones.append(
                        {
                            "price_range": [
                                float(price_bins[i]),
                                float(price_bins[i + 1]),
                            ],
                            "center": float(bin_centers[i]),
                            "type": "ask_dominance",
                            "ratio": float(ratio),
                            "strength": min(
                                10, float(inverse_ratio / threshold * 5)
                            ),  # Scale 0-10
                        }
                    )

            # Sort by imbalance strength
            imbalance_zones.sort(key=lambda x: x["strength"], reverse=True)

            # Track imbalance zones over time
            if len(self.orderbook_history) > 1:
                for zone in imbalance_zones:
                    zone["persistent"] = False
                    center = zone["center"]

                    # Check last orderbook
                    prev_orderbook = self.orderbook_history[-1]
                    if (
                        "liquidity" in prev_orderbook
                        and "imbalance" in prev_orderbook["liquidity"]
                    ):
                        prev_zones = prev_orderbook["liquidity"]["imbalance"].get(
                            "zones", []
                        )

                        for prev_zone in prev_zones:
                            prev_center = prev_zone.get("center", 0)
                            if abs(center - prev_center) / center < 0.01:  # Within 1%
                                zone["persistent"] = True
                                # If the imbalance is persistent, increase its strength
                                zone["strength"] = min(10, zone["strength"] * 1.2)
                                break

            # NEW: Calculate confidence scores
            for zone in imbalance_zones:
                # Base confidence on strength
                confidence = zone["strength"] * 0.7

                # Boost for persistent zones
                if zone.get("persistent", False):
                    confidence += 3

                # Cap at 10
                zone["confidence"] = min(10, confidence)

            # Filter out low-confidence zones
            imbalance_zones = [z for z in imbalance_zones if z.get("confidence", 0) > 5]

            return {"zones": imbalance_zones}

        except Exception as e:
            logger.error(f"Error in detecting imbalance zones: {e}")
            return {"zones": []}

    def _identify_advanced_liquidity_levels(self, orders, min_volume_factor=2.5):
        """Identifies advanced liquidity levels with cluster detection"""
        try:
            if not orders or len(orders) < 5:
                return []

            # Extract prices and volumes
            prices = np.array([order[0] for order in orders])
            volumes = np.array([order[1] for order in orders])

            # Calculate mean and standard deviation of volume
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)

            # Identify orders with anomalous volume (outliers)
            threshold = mean_volume + min_volume_factor * std_volume

            # Find indices of anomalous orders
            anomaly_indices = np.where(volumes > threshold)[0]

            # Create liquidity levels only for anomalous orders
            liquidity_levels = []

            for idx in anomaly_indices:
                price = prices[idx]
                volume = volumes[idx]

                # Calculate percentage of total volume
                volume_percentage = volume / np.sum(volumes) * 100

                # Calculate Z-score of volume to quantify the anomaly
                z_score = (volume - mean_volume) / std_volume if std_volume > 0 else 0

                # Relative strength on scale 0-10
                relative_strength = min(10, z_score)

                liquidity_levels.append(
                    {
                        "price": float(price),
                        "volume": float(volume),
                        "volume_percentage": float(volume_percentage),
                        "z_score": float(z_score),
                        "relative_strength": float(relative_strength),
                    }
                )

            # Sort by price
            liquidity_levels.sort(key=lambda x: x["price"])

            # Analyze related orders (clusters around anomalous orders)
            for level in liquidity_levels:
                level_price = level["price"]

                # Find orders within 0.5% of this level
                nearby_indices = np.where(
                    np.abs(prices - level_price) / level_price < 0.005
                )[0]

                if len(nearby_indices) > 1:  # At least one other order nearby
                    nearby_volume = np.sum(volumes[nearby_indices])
                    level["cluster_volume"] = float(nearby_volume)
                    level["cluster_order_count"] = int(len(nearby_indices))

                    # If there's a cluster, increase the strength
                    cluster_factor = min(2.0, 1 + (len(nearby_indices) / 10))
                    level["relative_strength"] = min(
                        10, level["relative_strength"] * cluster_factor
                    )

            # NEW: Apply advanced clustering to identify significant levels
            try:
                # Use DBSCAN for density-based clustering
                if len(prices) > 10:
                    X = prices.reshape(-1, 1)
                    clustering = DBSCAN(eps=0.002, min_samples=3).fit(X)

                    if hasattr(clustering, "labels_"):
                        labels = clustering.labels_
                        unique_labels = set(labels)

                        # Process clusters (ignore noise points labeled as -1)
                        clusters = []
                        for label in unique_labels:
                            if label == -1:  # Skip noise
                                continue

                            # Get points in this cluster
                            cluster_indices = np.where(labels == label)[0]
                            cluster_prices = prices[cluster_indices]
                            cluster_volumes = volumes[cluster_indices]

                            if len(cluster_prices) >= 3:  # Minimum cluster size
                                avg_price = np.mean(cluster_prices)
                                total_volume = np.sum(cluster_volumes)

                                # Calculate density (orders per price unit)
                                price_range = np.max(cluster_prices) - np.min(
                                    cluster_prices
                                )
                                density = len(cluster_prices) / (price_range + 0.00001)

                                # Add as a significant liquidity level if volume is significant
                                if total_volume > mean_volume * 3:
                                    clusters.append(
                                        {
                                            "price": float(avg_price),
                                            "volume": float(total_volume),
                                            "order_count": int(len(cluster_prices)),
                                            "density": float(density),
                                            "relative_strength": min(
                                                10, float(total_volume / mean_volume)
                                            ),
                                        }
                                    )

                        # Combine with individual anomalous levels
                        for cluster in clusters:
                            # Check if this cluster is already represented by an anomalous level
                            is_represented = False
                            for level in liquidity_levels:
                                if (
                                    abs(level["price"] - cluster["price"])
                                    / cluster["price"]
                                    < 0.005
                                ):
                                    is_represented = True
                                    # Update level with cluster information
                                    level["cluster_volume"] = cluster["volume"]
                                    level["cluster_order_count"] = cluster[
                                        "order_count"
                                    ]
                                    level["density"] = cluster.get("density", 0)
                                    # Take the higher strength
                                    level["relative_strength"] = max(
                                        level["relative_strength"],
                                        cluster["relative_strength"],
                                    )
                                    break

                            # Add new cluster if not already represented
                            if not is_represented:
                                liquidity_levels.append(
                                    {
                                        "price": cluster["price"],
                                        "volume": cluster["volume"],
                                        "volume_percentage": cluster["volume"]
                                        / np.sum(volumes)
                                        * 100,
                                        "relative_strength": cluster[
                                            "relative_strength"
                                        ],
                                        "cluster_order_count": cluster["order_count"],
                                        "density": cluster.get("density", 0),
                                    }
                                )
            except Exception as cluster_error:
                logger.warning(f"Error in advanced clustering: {cluster_error}")

            # Re-sort by strength
            liquidity_levels.sort(
                key=lambda x: x.get("relative_strength", 0), reverse=True
            )

            return liquidity_levels

        except Exception as e:
            logger.error(f"Error in identifying advanced liquidity levels: {e}")
            return []

    def _detect_advanced_order_walls(self, orders, min_strength=5.0):
        """Detects significant order walls with advanced analysis"""
        try:
            if not orders or len(orders) < 5:
                return []

            # Extract prices and volumes
            prices = np.array([order[0] for order in orders])
            volumes = np.array([order[1] for order in orders])

            # Calculate mean and standard deviation of volume
            mean_volume = np.mean(volumes)
            total_volume = np.sum(volumes)

            # Identify order walls (orders with volume much higher than average)
            walls = []

            for i, (price, volume) in enumerate(zip(prices, volumes)):
                # Calculate various metrics to evaluate wall strength
                volume_ratio = volume / mean_volume
                volume_percentage = (volume / total_volume) * 100
                price_impact = 0

                # Calculate price impact (how much volume is needed to overcome this wall)
                if i < len(orders) - 1:
                    price_diff = abs(prices[i + 1] - price)
                    price_impact = volume / price_diff if price_diff > 0 else volume

                # Combine metrics into a single wall "strength" score
                wall_strength = min(
                    10,
                    (
                        volume_ratio * 0.4
                        + volume_percentage * 0.3
                        + (price_impact / 1000) * 0.3
                    ),
                )

                # Include only significant walls
                if wall_strength >= min_strength:
                    walls.append(
                        [
                            float(price),
                            float(volume),
                            float(wall_strength),
                            float(volume_percentage),
                            float(volume_ratio),
                        ]
                    )

            # Sort by wall strength
            walls.sort(key=lambda x: x[2], reverse=True)

            # Analyze wall persistence over time
            if len(self.orderbook_history) > 1:
                persistent_walls = []

                for wall in walls:
                    price = wall[0]

                    # Check if this wall was in the previous orderbook
                    prev_orderbook = self.orderbook_history[-1]
                    if "liquidity" not in prev_orderbook:
                        continue

                    if (
                        "bid_walls" in prev_orderbook["liquidity"] and price < prices[0]
                    ):  # It's a bid
                        prev_walls = prev_orderbook["liquidity"]["bid_walls"]
                        for prev_wall in prev_walls:
                            if (
                                len(prev_wall) > 0
                                and abs(prev_wall[0] - price) / price < 0.005
                            ):  # Within 0.5%
                                # This is a persistent wall
                                persistent_walls.append(wall)
                                # Increase strength for persistence
                                wall[2] = min(10, wall[2] * 1.2)
                                break
                    elif (
                        "ask_walls" in prev_orderbook["liquidity"] and price > prices[0]
                    ):  # It's an ask
                        prev_walls = prev_orderbook["liquidity"]["ask_walls"]
                        for prev_wall in prev_walls:
                            if (
                                len(prev_wall) > 0
                                and abs(prev_wall[0] - price) / price < 0.005
                            ):  # Within 0.5%
                                # This is a persistent wall
                                persistent_walls.append(wall)
                                # Increase strength for persistence
                                wall[2] = min(10, wall[2] * 1.2)
                                break

                # Re-sort after persistence adjustments
                walls.sort(key=lambda x: x[2], reverse=True)

            # NEW: Advanced analysis of wall effectiveness
            # Check if similar walls were broken in recent history
            if len(self.orderbook_history) > 3 and len(self.trades_history) > 0:
                for wall in walls:
                    wall_price = wall[0]
                    wall_effectiveness = 1.0  # Base effectiveness

                    # Get recent trades
                    recent_trades = list(self.trades_history)

                    # Check if any trades crossed this wall price
                    trades_crossing_wall = []
                    for trade in recent_trades:
                        if "price" in trade and "side" in trade:
                            trade_price = trade["price"]
                            trade_side = trade["side"]

                            # For bid walls, check sells below the wall
                            if (
                                wall_price < prices[0]
                                and trade_side == "sell"
                                and trade_price < wall_price
                            ):
                                trades_crossing_wall.append(trade)

                            # For ask walls, check buys above the wall
                            elif (
                                wall_price > prices[0]
                                and trade_side == "buy"
                                and trade_price > wall_price
                            ):
                                trades_crossing_wall.append(trade)

                    # If wall was crossed, reduce effectiveness
                    if trades_crossing_wall:
                        # Reduce effectiveness based on how many trades crossed the wall
                        effectiveness_reduction = min(
                            0.8, len(trades_crossing_wall) / 10
                        )
                        wall_effectiveness = max(
                            0.2, wall_effectiveness - effectiveness_reduction
                        )

                        # Append effectiveness to the wall data
                        if len(wall) <= 5:
                            wall.append(float(wall_effectiveness))
                        else:
                            wall[5] = float(wall_effectiveness)

                        # Adjust wall strength based on effectiveness
                        wall[2] = wall[2] * wall_effectiveness

            return walls

        except Exception as e:
            logger.error(f"Error in detecting order walls: {e}")
            return []

    def _analyze_liquidity_changes(self, current_orderbook):
        """Analyzes liquidity changes between orderbooks to detect accumulation/distribution"""
        try:
            # If we don't have previous orderbooks, we can't calculate changes
            if len(self.orderbook_history) < 2:
                return {"status": "insufficient_history"}

            # Get the previous orderbook for comparison
            previous_orderbook = None
            for ob in reversed(list(self.orderbook_history)[:-1]):
                if "raw" in ob:
                    previous_orderbook = ob
                    break

            if not previous_orderbook:
                return {"status": "no_previous_orderbook"}
                
            # Make sure current_orderbook has the 'raw' field
            if "raw" not in current_orderbook:
                logger.warning("Current orderbook missing 'raw' field")
                return {"status": "missing_raw_data"}

            # Extract times to calculate time delta
            current_time = current_orderbook["timestamp"]
            previous_time = previous_orderbook["timestamp"]
            time_delta = (current_time - previous_time).total_seconds()

            if time_delta <= 0:
                return {"status": "invalid_time_delta"}

            # Extract metrics for comparison
            current_bid_volume = current_orderbook.get("metrics", {}).get(
                "bid_volume", 0
            )
            current_ask_volume = current_orderbook.get("metrics", {}).get(
                "ask_volume", 0
            )
            previous_bid_volume = previous_orderbook.get("metrics", {}).get(
                "bid_volume", 0
            )
            previous_ask_volume = previous_orderbook.get("metrics", {}).get(
                "ask_volume", 0
            )

            # Calculate changes
            bid_volume_change = current_bid_volume - previous_bid_volume
            ask_volume_change = current_ask_volume - previous_ask_volume
            bid_volume_change_percent = (
                (bid_volume_change / previous_bid_volume * 100)
                if previous_bid_volume > 0
                else 0
            )
            ask_volume_change_percent = (
                (ask_volume_change / previous_ask_volume * 100)
                if previous_ask_volume > 0
                else 0
            )

            # Calculate rate of change per second
            bid_volume_change_rate = bid_volume_change / time_delta
            ask_volume_change_rate = ask_volume_change / time_delta

            # Analysis of imbalance in volume change
            accumulation = None
            accumulation_strength = 0

            if abs(bid_volume_change) > 0 or abs(ask_volume_change) > 0:
                total_volume_change = abs(bid_volume_change) + abs(ask_volume_change)
                bid_change_proportion = (
                    abs(bid_volume_change) / total_volume_change
                    if total_volume_change > 0
                    else 0.5
                )

                # Determine if there's accumulation or distribution
                if bid_volume_change > 0 and ask_volume_change < 0:
                    accumulation = True  # More buying, less selling = accumulation
                    accumulation_strength = min(
                        10,
                        (bid_volume_change_percent + abs(ask_volume_change_percent))
                        / 10,
                    )
                elif bid_volume_change < 0 and ask_volume_change > 0:
                    accumulation = False  # Less buying, more selling = distribution
                    accumulation_strength = min(
                        10,
                        (abs(bid_volume_change_percent) + ask_volume_change_percent)
                        / 10,
                    )
                else:
                    # Mixed direction, determine based on imbalance
                    if bid_change_proportion > 0.6:  # Greater change on bid side
                        accumulation = (
                            bid_volume_change > 0
                        )  # accumulation if bids are increasing
                        accumulation_strength = min(
                            10, abs(bid_volume_change_percent) / 5
                        )
                    elif bid_change_proportion < 0.4:  # Greater change on ask side
                        accumulation = (
                            ask_volume_change < 0
                        )  # accumulation if asks are decreasing
                        accumulation_strength = min(
                            10, abs(ask_volume_change_percent) / 5
                        )
                    else:
                        accumulation = None  # Non-directional change
                        accumulation_strength = 0

            # Analysis of change in order walls
            # Compare liquidity walls between orderbooks
            current_bid_walls = current_orderbook.get("liquidity", {}).get(
                "bid_walls", []
            )
            current_ask_walls = current_orderbook.get("liquidity", {}).get(
                "ask_walls", []
            )
            previous_bid_walls = previous_orderbook.get("liquidity", {}).get(
                "bid_walls", []
            )
            previous_ask_walls = previous_orderbook.get("liquidity", {}).get(
                "ask_walls", []
            )

            # Analysis of new or removed walls
            new_significant_walls = []
            removed_significant_walls = []

            # Check for new significant walls in bids
            for wall in current_bid_walls:
                if len(wall) >= 3 and wall[2] >= 7:  # High strength
                    found = False
                    for prev_wall in previous_bid_walls:
                        if (
                            len(prev_wall) >= 1
                            and abs(wall[0] - prev_wall[0]) / prev_wall[0] < 0.001
                        ):  # Same price
                            found = True
                            break
                    if not found:
                        new_significant_walls.append(
                            {
                                "price": float(wall[0]),
                                "type": "bid",
                                "strength": float(wall[2]),
                            }
                        )

            # Check for new significant walls in asks
            for wall in current_ask_walls:
                if len(wall) >= 3 and wall[2] >= 7:  # High strength
                    found = False
                    for prev_wall in previous_ask_walls:
                        if (
                            len(prev_wall) >= 1
                            and abs(wall[0] - prev_wall[0]) / prev_wall[0] < 0.001
                        ):  # Same price
                            found = True
                            break
                    if not found:
                        new_significant_walls.append(
                            {
                                "price": float(wall[0]),
                                "type": "ask",
                                "strength": float(wall[2]),
                            }
                        )

            # Check for removed walls in bids
            for prev_wall in previous_bid_walls:
                if len(prev_wall) >= 3 and prev_wall[2] >= 7:  # High strength
                    found = False
                    for wall in current_bid_walls:
                        if (
                            len(wall) >= 1
                            and abs(wall[0] - prev_wall[0]) / prev_wall[0] < 0.001
                        ):  # Same price
                            found = True
                            break
                    if not found:
                        removed_significant_walls.append(
                            {
                                "price": float(prev_wall[0]),
                                "type": "bid",
                                "strength": float(prev_wall[2]),
                            }
                        )

            # Check for removed walls in asks
            for prev_wall in previous_ask_walls:
                if len(prev_wall) >= 3 and prev_wall[2] >= 7:  # High strength
                    found = False
                    for wall in current_ask_walls:
                        if (
                            len(wall) >= 1
                            and abs(wall[0] - prev_wall[0]) / prev_wall[0] < 0.001
                        ):  # Same price
                            found = True
                            break
                    if not found:
                        removed_significant_walls.append(
                            {
                                "price": float(prev_wall[0]),
                                "type": "ask",
                                "strength": float(prev_wall[2]),
                            }
                        )

            # Analyze price-level liquidity changes
            # This helps detect stealth accumulation/distribution at specific levels
            price_level_changes = []

            # Define price levels (10 levels around mid price)
            try:
                if "raw" in current_orderbook and "bids" in current_orderbook["raw"] and len(current_orderbook["raw"]["bids"]) > 0 and "asks" in current_orderbook["raw"] and len(current_orderbook["raw"]["asks"]) > 0:
                    current_mid = (
                        current_orderbook["raw"]["bids"][0][0]
                        + current_orderbook["raw"]["asks"][0][0]
                    ) / 2
                else:
                    # Fallback: usa il Current price o recente
                    current_mid = self.data.iloc[-1]['close'] if len(self.data) > 0 else 0
                    logger.info(f"using fallback price per liquidity analysis: {current_mid}")
            except Exception as e:
                logger.error(f"Error nell'accesso ai dati dell'orderbook: {e}")
                return {"status": "error", "message": str(e)}
                
            price_range = 0.02  # 2% range
            lower_bound = current_mid * (1 - price_range)
            upper_bound = current_mid * (1 + price_range)

            # Create price level bins
            level_count = 10
            level_bins = np.linspace(lower_bound, upper_bound, level_count + 1)

            # Analyze current and previous orderbooks at these levels
            for i in range(level_count):
                level_min = level_bins[i]
                level_max = level_bins[i + 1]
                level_center = (level_min + level_max) / 2

                # Current orderbook liquidity at this level
                current_bid_vol = sum(
                    [
                        o[1]
                        for o in current_orderbook["raw"]["bids"]
                        if level_min <= o[0] < level_max
                    ]
                )
                current_ask_vol = sum(
                    [
                        o[1]
                        for o in current_orderbook["raw"]["asks"]
                        if level_min <= o[0] < level_max
                    ]
                )

                # Previous orderbook liquidity at this level
                prev_bid_vol = sum(
                    [
                        o[1]
                        for o in previous_orderbook["raw"]["bids"]
                        if level_min <= o[0] < level_max
                    ]
                )
                prev_ask_vol = sum(
                    [
                        o[1]
                        for o in previous_orderbook["raw"]["asks"]
                        if level_min <= o[0] < level_max
                    ]
                )

                # Calculate changes
                bid_change = current_bid_vol - prev_bid_vol
                ask_change = current_ask_vol - prev_ask_vol

                # Only report significant changes
                if abs(bid_change) > 0 or abs(ask_change) > 0:
                    # Calculate change percentage
                    bid_change_pct = (
                        (bid_change / prev_bid_vol * 100) if prev_bid_vol > 0 else 0
                    )
                    ask_change_pct = (
                        (ask_change / prev_ask_vol * 100) if prev_ask_vol > 0 else 0
                    )

                    # Identify significant changes (>20%)
                    if abs(bid_change_pct) > 20 or abs(ask_change_pct) > 20:
                        price_level_changes.append(
                            {
                                "price_level": float(level_center),
                                "price_range": [float(level_min), float(level_max)],
                                "bid_change": float(bid_change),
                                "ask_change": float(ask_change),
                                "bid_change_pct": float(bid_change_pct),
                                "ask_change_pct": float(ask_change_pct),
                                "accumulated": bid_change > 0 and ask_change < 0,
                                "distributed": bid_change < 0 and ask_change > 0,
                            }
                        )

            # NEW: Calculate orderbook depth change
            # Sum volume at different depths from mid price
            depth_levels = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5% away from mid
            current_depth = {"bid": {}, "ask": {}}
            previous_depth = {"bid": {}, "ask": {}}

            for level in depth_levels:
                # Current depths
                current_depth["bid"][level] = sum(
                    [
                        o[1]
                        for o in current_orderbook["raw"]["bids"]
                        if current_mid * (1 - level) <= o[0] <= current_mid
                    ]
                )
                current_depth["ask"][level] = sum(
                    [
                        o[1]
                        for o in current_orderbook["raw"]["asks"]
                        if current_mid <= o[0] <= current_mid * (1 + level)
                    ]
                )

                # Previous depths
                prev_mid = (
                    previous_orderbook["raw"]["bids"][0][0]
                    + previous_orderbook["raw"]["asks"][0][0]
                ) / 2
                previous_depth["bid"][level] = sum(
                    [
                        o[1]
                        for o in previous_orderbook["raw"]["bids"]
                        if prev_mid * (1 - level) <= o[0] <= prev_mid
                    ]
                )
                previous_depth["ask"][level] = sum(
                    [
                        o[1]
                        for o in previous_orderbook["raw"]["asks"]
                        if prev_mid <= o[0] <= prev_mid * (1 + level)
                    ]
                )

            # Calculate changes for each depth level
            depth_changes = {"bid": {}, "ask": {}}
            for level in depth_levels:
                depth_changes["bid"][level] = {
                    "absolute": current_depth["bid"][level]
                    - previous_depth["bid"][level],
                    "percent": (
                        (
                            (current_depth["bid"][level] / previous_depth["bid"][level])
                            - 1
                        )
                        * 100
                        if previous_depth["bid"][level] > 0
                        else 0
                    ),
                }
                depth_changes["ask"][level] = {
                    "absolute": current_depth["ask"][level]
                    - previous_depth["ask"][level],
                    "percent": (
                        (
                            (current_depth["ask"][level] / previous_depth["ask"][level])
                            - 1
                        )
                        * 100
                        if previous_depth["ask"][level] > 0
                        else 0
                    ),
                }

            # Analysis result
            result = {
                "time_delta": float(time_delta),
                "bid_volume_change": float(bid_volume_change),
                "ask_volume_change": float(ask_volume_change),
                "bid_volume_change_percent": float(bid_volume_change_percent),
                "ask_volume_change_percent": float(ask_volume_change_percent),
                "bid_volume_change_rate": float(bid_volume_change_rate),
                "ask_volume_change_rate": float(ask_volume_change_rate),
                "accumulation": accumulation,
                "accumulation_strength": (
                    float(accumulation_strength) if accumulation is not None else 0
                ),
                "new_significant_walls": new_significant_walls,
                "removed_significant_walls": removed_significant_walls,
                "price_level_changes": price_level_changes,
                "depth_changes": depth_changes,
            }

            return result
        except Exception as e:
            logger.error(f"Error in analyzing liquidity changes: {e}")
            return {"status": "error", "reason": str(e)}

    def _analyze_order_footprint(self, orderbook):
        """Analyzes the orderbook 'Footprint' to detect manipulation or anomalous behaviors"""
        try:
            # Extract bids and asks
            bids = orderbook["bids"]
            asks = orderbook["asks"]

            if not bids or not asks:
                return {"status": "empty_orderbook"}

            # Extract prices and volumes
            bid_prices = [order[0] for order in bids]
            bid_volumes = [order[1] for order in bids]
            ask_prices = [order[0] for order in asks]
            ask_volumes = [order[1] for order in asks]

            # Calculate average order size
            avg_bid_size = np.mean(bid_volumes) if bid_volumes else 0
            avg_ask_size = np.mean(ask_volumes) if ask_volumes else 0

            # Calculate standard deviation of order sizes
            std_bid_size = np.std(bid_volumes) if len(bid_volumes) > 1 else 0
            std_ask_size = np.std(ask_volumes) if len(ask_volumes) > 1 else 0

            # Calculate median of order sizes
            median_bid_size = np.median(bid_volumes) if bid_volumes else 0
            median_ask_size = np.median(ask_volumes) if ask_volumes else 0

            # Calculate ratio between mean and median (indicator of skew)
            bid_skew = avg_bid_size / median_bid_size if median_bid_size > 0 else 1
            ask_skew = avg_ask_size / median_ask_size if median_ask_size > 0 else 1

            # Identify "spoofing" signals (large orders far from mid price that could be manipulative)
            spoofing_signals = []

            # Calculate mid price
            best_bid = bid_prices[0] if bid_prices else 0
            best_ask = ask_prices[0] if ask_prices else 0
            mid_price = (best_bid + best_ask) / 2

            # Check for large orders in bids far from price
            for i, (price, volume) in enumerate(zip(bid_prices, bid_volumes)):
                if (
                    i > 0
                    and volume > avg_bid_size * 5
                    and (mid_price - price) / mid_price > 0.01
                ):
                    # Large order, far from mid price (>1%)
                    spoofing_signals.append(
                        {
                            "type": "large_distant_bid",
                            "price": float(price),
                            "volume": float(volume),
                            "distance_percent": float(
                                (mid_price - price) / mid_price * 100
                            ),
                            "size_ratio": float(volume / avg_bid_size),
                        }
                    )

            # Check for large orders in asks far from price
            for i, (price, volume) in enumerate(zip(ask_prices, ask_volumes)):
                if (
                    i > 0
                    and volume > avg_ask_size * 5
                    and (price - mid_price) / mid_price > 0.01
                ):
                    # Large order, far from mid price (>1%)
                    spoofing_signals.append(
                        {
                            "type": "large_distant_ask",
                            "price": float(price),
                            "volume": float(volume),
                            "distance_percent": float(
                                (price - mid_price) / mid_price * 100
                            ),
                            "size_ratio": float(volume / avg_ask_size),
                        }
                    )

            # Determine if there is probable manipulation in the orderbook
            probable_manipulation = len(spoofing_signals) > 0

            # Analyze order distribution to detect patterns
            # Calculate percentage of volume in the top 10 price levels
            top_10_bid_volume = (
                sum(bid_volumes[:10]) if len(bid_volumes) >= 10 else sum(bid_volumes)
            )
            top_10_ask_volume = (
                sum(ask_volumes[:10]) if len(ask_volumes) >= 10 else sum(ask_volumes)
            )

            total_bid_volume = sum(bid_volumes)
            total_ask_volume = sum(ask_volumes)

            top_10_bid_percentage = (
                (top_10_bid_volume / total_bid_volume * 100)
                if total_bid_volume > 0
                else 0
            )
            top_10_ask_percentage = (
                (top_10_ask_volume / total_ask_volume * 100)
                if total_ask_volume > 0
                else 0
            )

            # Detect "iceberg" pattern (hidden large orders)
            iceberg_pattern = False
            iceberg_side = None

            # An iceberg pattern often has anomalous volumes near the mid price
            if avg_bid_size > 0 and bid_volumes[0] / avg_bid_size < 0.3:
                iceberg_pattern = True
                iceberg_side = "bid"

            if avg_ask_size > 0 and ask_volumes[0] / avg_ask_size < 0.3:
                iceberg_pattern = True
                iceberg_side = "ask" if iceberg_side is None else "both"

            # Analyze the footprint for strategic placement patterns
            strategic_patterns = []

            # Pattern 1: Step formation (gradually increasing/decreasing sizes)
            if len(bid_volumes) > 5:
                diffs = np.diff(bid_volumes[:5])
                if np.all(diffs > 0):
                    strategic_patterns.append(
                        {
                            "type": "ascending_bid_sizes",
                            "description": "Gradually increasing bid sizes near the spread",
                            "interpretation": "Strategic accumulation",
                        }
                    )
                elif np.all(diffs < 0):
                    strategic_patterns.append(
                        {
                            "type": "descending_bid_sizes",
                            "description": "Gradually decreasing bid sizes near the spread",
                            "interpretation": "Protective positioning",
                        }
                    )

            if len(ask_volumes) > 5:
                diffs = np.diff(ask_volumes[:5])
                if np.all(diffs > 0):
                    strategic_patterns.append(
                        {
                            "type": "ascending_ask_sizes",
                            "description": "Gradually increasing ask sizes near the spread",
                            "interpretation": "Protective selling",
                        }
                    )
                elif np.all(diffs < 0):
                    strategic_patterns.append(
                        {
                            "type": "descending_ask_sizes",
                            "description": "Gradually decreasing ask sizes near the spread",
                            "interpretation": "Strategic distribution",
                        }
                    )

            # Pattern 2: Fence posts (alternating large and small orders)
            if len(bid_volumes) > 6:
                alternating = True
                for i in range(1, 6, 2):
                    if not (
                        bid_volumes[i] > bid_volumes[i - 1] * 2
                        and bid_volumes[i] > bid_volumes[i + 1] * 2
                    ):
                        alternating = False
                        break
                if alternating:
                    strategic_patterns.append(
                        {
                            "type": "bid_fence_posts",
                            "description": "Alternating large and small bid orders",
                            "interpretation": "Advanced accumulation strategy",
                        }
                    )

            if len(ask_volumes) > 6:
                alternating = True
                for i in range(1, 6, 2):
                    if not (
                        ask_volumes[i] > ask_volumes[i - 1] * 2
                        and ask_volumes[i] > ask_volumes[i + 1] * 2
                    ):
                        alternating = False
                        break
                if alternating:
                    strategic_patterns.append(
                        {
                            "type": "ask_fence_posts",
                            "description": "Alternating large and small ask orders",
                            "interpretation": "Advanced distribution strategy",
                        }
                    )

            # NEW: Pattern 3: Clustering analysis (DBSCAN)
            # Look for unusual clusters in order volume distribution
            try:
                # Prepare data for clustering
                X_bid = np.column_stack([np.array(bid_prices), np.array(bid_volumes)])
                X_ask = np.column_stack([np.array(ask_prices), np.array(ask_volumes)])

                # Normalize the data for better clustering
                scaler_bid = StandardScaler()
                scaler_ask = StandardScaler()

                # Only perform clustering if we have enough data
                if len(X_bid) >= 5:
                    X_bid_scaled = scaler_bid.fit_transform(X_bid)

                    # Apply DBSCAN
                    clustering_bid = DBSCAN(eps=0.5, min_samples=2).fit(X_bid_scaled)
                    labels_bid = clustering_bid.labels_

                    # Identify clusters
                    unique_labels_bid = set(labels_bid)
                    for label in unique_labels_bid:
                        if label != -1:  # -1 is noise
                            mask = labels_bid == label
                            cluster_size = np.sum(mask)

                            if cluster_size >= 3:  # At least 3 points to be meaningful
                                average_volume = np.mean(X_bid[mask, 1])
                                average_price = np.mean(X_bid[mask, 0])

                                # Check if this is an unusual cluster
                                if average_volume > avg_bid_size * 2:
                                    strategic_patterns.append(
                                        {
                                            "type": "bid_volume_cluster",
                                            "price": float(average_price),
                                            "average_volume": float(average_volume),
                                            "points_in_cluster": int(cluster_size),
                                            "interpretation": "Potential bid accumulation zone",
                                        }
                                    )

                if len(X_ask) >= 5:
                    X_ask_scaled = scaler_ask.fit_transform(X_ask)

                    # Apply DBSCAN
                    clustering_ask = DBSCAN(eps=0.5, min_samples=2).fit(X_ask_scaled)
                    labels_ask = clustering_ask.labels_

                    # Identify clusters
                    unique_labels_ask = set(labels_ask)
                    for label in unique_labels_ask:
                        if label != -1:  # -1 is noise
                            mask = labels_ask == label
                            cluster_size = np.sum(mask)

                            if cluster_size >= 3:  # At least 3 points to be meaningful
                                average_volume = np.mean(X_ask[mask, 1])
                                average_price = np.mean(X_ask[mask, 0])

                                # Check if this is an unusual cluster
                                if average_volume > avg_ask_size * 2:
                                    strategic_patterns.append(
                                        {
                                            "type": "ask_volume_cluster",
                                            "price": float(average_price),
                                            "average_volume": float(average_volume),
                                            "points_in_cluster": int(cluster_size),
                                            "interpretation": "Potential ask distribution zone",
                                        }
                                    )
            except Exception as e:
                logger.warning(f"Error in cluster analysis: {e}")

            # Analysis result
            footprint = {
                "avg_bid_size": float(avg_bid_size),
                "avg_ask_size": float(avg_ask_size),
                "median_bid_size": float(median_bid_size),
                "median_ask_size": float(median_ask_size),
                "bid_skew": float(bid_skew),
                "ask_skew": float(ask_skew),
                "top_10_bid_percentage": float(top_10_bid_percentage),
                "top_10_ask_percentage": float(top_10_ask_percentage),
                "spoofing_signals": spoofing_signals,
                "probable_manipulation": probable_manipulation,
                "iceberg_pattern": iceberg_pattern,
                "iceberg_side": iceberg_side,
                "strategic_patterns": strategic_patterns,
            }

            return footprint
        except Exception as e:
            logger.error(f"Error in analyzing orderbook footprint: {e}")
            return {"status": "error", "reason": str(e)}

    def _detect_manipulation_signals(self, orderbook):
        """
        NEW: Detect potential market manipulation in the orderbook
        """
        try:
            if not orderbook or not orderbook["bids"] or not orderbook["asks"]:
                return {"detected": False}

            # Extract prices and volumes
            bid_prices = np.array([order[0] for order in orderbook["bids"]])
            bid_volumes = np.array([order[1] for order in orderbook["bids"]])
            ask_prices = np.array([order[0] for order in orderbook["asks"]])
            ask_volumes = np.array([order[1] for order in orderbook["asks"]])

            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2

            manipulation_signals = {
                "detected": False,
                "type": None,
                "confidence": 0,
                "details": {},
            }

            # 1. Detect layering (multiple orders on one side to create false impression)
            if len(bid_prices) > 10:
                bid_clusters = self._identify_price_clusters(
                    bid_prices[:10], bid_volumes[:10]
                )
                if (
                    bid_clusters
                    and max(cluster["density"] for cluster in bid_clusters) > 3
                ):
                    densest_cluster = max(bid_clusters, key=lambda x: x["density"])
                    if densest_cluster["density"] > 3:
                        manipulation_signals["detected"] = True
                        manipulation_signals["type"] = "bid_layering"
                        manipulation_signals["confidence"] = min(
                            10, densest_cluster["density"]
                        )
                        manipulation_signals["details"]["location"] = densest_cluster[
                            "center"
                        ]
                        manipulation_signals["details"]["density"] = densest_cluster[
                            "density"
                        ]

            if len(ask_prices) > 10:
                ask_clusters = self._identify_price_clusters(
                    ask_prices[:10], ask_volumes[:10]
                )
                if (
                    ask_clusters
                    and max(cluster["density"] for cluster in ask_clusters) > 3
                ):
                    densest_cluster = max(ask_clusters, key=lambda x: x["density"])
                    if densest_cluster["density"] > 3:
                        manipulation_signals["detected"] = True
                        manipulation_signals["type"] = "ask_layering"
                        manipulation_signals["confidence"] = min(
                            10, densest_cluster["density"]
                        )
                        manipulation_signals["details"]["location"] = densest_cluster[
                            "center"
                        ]
                        manipulation_signals["details"]["density"] = densest_cluster[
                            "density"
                        ]

            # 2. Detect spoofing (large orders far from mid price)
            far_bid_idx = np.where(bid_prices < mid_price * 0.98)[0]
            if len(far_bid_idx) > 0:
                far_bid_volumes = bid_volumes[far_bid_idx]
                if np.any(far_bid_volumes > np.mean(bid_volumes) * 5):
                    largest_far_idx = far_bid_idx[np.argmax(far_bid_volumes)]
                    manipulation_signals["detected"] = True
                    manipulation_signals["type"] = "bid_spoofing"
                    manipulation_signals["confidence"] = min(
                        10, far_bid_volumes.max() / np.mean(bid_volumes)
                    )
                    manipulation_signals["details"]["location"] = float(
                        bid_prices[largest_far_idx]
                    )
                    manipulation_signals["details"]["volume"] = float(
                        bid_volumes[largest_far_idx]
                    )

            far_ask_idx = np.where(ask_prices > mid_price * 1.02)[0]
            if len(far_ask_idx) > 0:
                far_ask_volumes = ask_volumes[far_ask_idx]
                if np.any(far_ask_volumes > np.mean(ask_volumes) * 5):
                    largest_far_idx = far_ask_idx[np.argmax(far_ask_volumes)]
                    manipulation_signals["detected"] = True
                    manipulation_signals["type"] = "ask_spoofing"
                    manipulation_signals["confidence"] = min(
                        10, far_ask_volumes.max() / np.mean(ask_volumes)
                    )
                    manipulation_signals["details"]["location"] = float(
                        ask_prices[largest_far_idx]
                    )
                    manipulation_signals["details"]["volume"] = float(
                        ask_volumes[largest_far_idx]
                    )

            # 3. Detect momentum ignition (rapid order placement/cancellation)
            # This would require comparing with historical orderbooks
            if len(self.orderbook_history) > 2:
                previous_orderbook = self.orderbook_history[-2]
                if "raw" in previous_orderbook:
                    prev_best_bid = (
                        previous_orderbook["raw"]["bids"][0][0]
                        if previous_orderbook["raw"]["bids"]
                        else 0
                    )
                    prev_best_ask = (
                        previous_orderbook["raw"]["asks"][0][0]
                        if previous_orderbook["raw"]["asks"]
                        else 0
                    )

                    # Large ask pressure appeared then disappeared
                    if prev_best_ask - prev_best_bid > (best_ask - best_bid) * 2:
                        manipulation_signals["detected"] = True
                        manipulation_signals["type"] = "momentum_ignition"
                        manipulation_signals["confidence"] = 7
                        manipulation_signals["details"]["spread_change"] = float(
                            (prev_best_ask - prev_best_bid) / (best_ask - best_bid)
                        )

            # If detection is successful, add timestamp
            if manipulation_signals["detected"]:
                manipulation_signals["timestamp"] = datetime.now()

            return manipulation_signals

        except Exception as e:
            logger.error(f"Error in detecting manipulation signals: {e}")
            return {"detected": False}

    def _analyze_order_clusters(self, orderbook):
        """
        NEW: Analyzes clusters of orders in the orderbook to identify significant levels
        """
        try:
            if not orderbook or not orderbook["bids"] or not orderbook["asks"]:
                return {"clusters": []}

            # Extract prices and volumes
            bid_prices = np.array([order[0] for order in orderbook["bids"]])
            bid_volumes = np.array([order[1] for order in orderbook["bids"]])
            ask_prices = np.array([order[0] for order in orderbook["asks"]])
            ask_volumes = np.array([order[1] for order in orderbook["asks"]])

            # Combine all data for PCA
            all_prices = np.concatenate([bid_prices, ask_prices])
            all_volumes = np.concatenate([bid_volumes, ask_volumes])
            sides = np.concatenate(
                [np.zeros(len(bid_prices)), np.ones(len(ask_prices))]
            )

            # Create feature matrix
            X = np.column_stack([all_prices, all_volumes])

            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Apply DBSCAN to identify clusters
            clustering = DBSCAN(eps=0.3, min_samples=3).fit(X_pca)
            labels = clustering.labels_

            # Analyze clusters
            clusters = []
            for label in set(labels):
                if label == -1:  # Skip noise points
                    continue

                mask = labels == label
                cluster_prices = all_prices[mask]
                cluster_volumes = all_volumes[mask]
                cluster_sides = sides[mask]

                # Calculate average points
                avg_price = np.mean(cluster_prices)
                total_volume = np.sum(cluster_volumes)

                # Determine if the cluster is predominantly bids or asks
                bid_ratio = np.sum(cluster_sides == 0) / len(cluster_sides)

                cluster_type = (
                    "support"
                    if bid_ratio > 0.7
                    else "resistance" if bid_ratio < 0.3 else "mixed"
                )

                # Calculate cluster strength (relative volume)
                relative_strength = (
                    total_volume / np.mean(all_volumes) / len(cluster_prices)
                )

                # Calculate density
                price_range = np.max(cluster_prices) - np.min(cluster_prices)
                density = len(cluster_prices) / (
                    price_range + 0.00001
                )  # Avoid division by zero

                # Add to clusters if significant
                if len(cluster_prices) >= 3 and relative_strength > 1.5:
                    clusters.append(
                        {
                            "price": float(avg_price),
                            "type": cluster_type,
                            "volume": float(total_volume),
                            "order_count": int(len(cluster_prices)),
                            "price_range": [
                                float(np.min(cluster_prices)),
                                float(np.max(cluster_prices)),
                            ],
                            "density": float(density),
                            "relative_strength": float(min(10, relative_strength)),
                        }
                    )

            # Sort by strength
            clusters.sort(key=lambda x: x["relative_strength"], reverse=True)

            return {"clusters": clusters, "count": len(clusters)}

        except Exception as e:
            logger.error(f"Error in analyzing order clusters: {e}")
            return {"clusters": []}

    def _detect_stop_hunting_zones(self, orderbook):
        """
        NEW: Identifies potential stop hunting zones in the orderbook
        """
        try:
            if not orderbook or not orderbook["bids"] or not orderbook["asks"]:
                return {"zones": []}

            # Extract prices and volumes
            bid_prices = np.array([order[0] for order in orderbook["bids"]])
            bid_volumes = np.array([order[1] for order in orderbook["bids"]])
            ask_prices = np.array([order[0] for order in orderbook["asks"]])
            ask_volumes = np.array([order[1] for order in orderbook["asks"]])

            mid_price = (bid_prices[0] + ask_prices[0]) / 2

            # Define potential stop loss levels
            stop_hunting_zones = []

            # Look for unusual volume concentrations below key psychological levels
            psychological_levels = []

            # Generate psychological levels based on current price
            base_price = int(mid_price)

            # Round numbers
            for level in [base_price - 50, base_price, base_price + 50]:
                psychological_levels.append(level)

            # Look for volume concentrations just beyond these levels
            for level in psychological_levels:
                # Check for bid concentration just below level (short stop hunting)
                close_bids_mask = (bid_prices < level) & (bid_prices > level * 0.99)
                if np.any(close_bids_mask):
                    close_bids_volume = np.sum(bid_volumes[close_bids_mask])
                    avg_bid_volume = np.mean(bid_volumes)

                    if close_bids_volume > avg_bid_volume * 3:
                        stop_hunting_zones.append(
                            {
                                "level": float(level),
                                "type": "short_stops",
                                "price_range": [
                                    float(np.min(bid_prices[close_bids_mask])),
                                    float(np.max(bid_prices[close_bids_mask])),
                                ],
                                "total_volume": float(close_bids_volume),
                                "relative_volume": float(
                                    close_bids_volume / avg_bid_volume
                                ),
                                "confidence": min(
                                    10, float(close_bids_volume / avg_bid_volume / 2)
                                ),
                            }
                        )

                # Check for ask concentration just above level (long stop hunting)
                close_asks_mask = (ask_prices > level) & (ask_prices < level * 1.01)
                if np.any(close_asks_mask):
                    close_asks_volume = np.sum(ask_volumes[close_asks_mask])
                    avg_ask_volume = np.mean(ask_volumes)

                    if close_asks_volume > avg_ask_volume * 3:
                        stop_hunting_zones.append(
                            {
                                "level": float(level),
                                "type": "long_stops",
                                "price_range": [
                                    float(np.min(ask_prices[close_asks_mask])),
                                    float(np.max(ask_prices[close_asks_mask])),
                                ],
                                "total_volume": float(close_asks_volume),
                                "relative_volume": float(
                                    close_asks_volume / avg_ask_volume
                                ),
                                "confidence": min(
                                    10, float(close_asks_volume / avg_ask_volume / 2)
                                ),
                            }
                        )

            # Look for thin liquidity zones that might be stop hunting targets
            if len(bid_prices) > 10:
                bid_gaps = bid_prices[:-1] - bid_prices[1:]
                large_gap_indices = np.where(bid_gaps > np.mean(bid_gaps) * 3)[0]

                for idx in large_gap_indices:
                    upper_price = bid_prices[idx]
                    lower_price = bid_prices[idx + 1]
                    gap_size = upper_price - lower_price

                    stop_hunting_zones.append(
                        {
                            "level": float(lower_price),
                            "type": "thin_liquidity_bid",
                            "price_range": [float(lower_price), float(upper_price)],
                            "gap_size": float(gap_size),
                            "relative_gap": float(gap_size / np.mean(bid_gaps)),
                            "confidence": min(
                                10, float(gap_size / np.mean(bid_gaps) / 2)
                            ),
                        }
                    )

            if len(ask_prices) > 10:
                ask_gaps = ask_prices[1:] - ask_prices[:-1]
                large_gap_indices = np.where(ask_gaps > np.mean(ask_gaps) * 3)[0]

                for idx in large_gap_indices:
                    lower_price = ask_prices[idx]
                    upper_price = ask_prices[idx + 1]
                    gap_size = upper_price - lower_price

                    stop_hunting_zones.append(
                        {
                            "level": float(upper_price),
                            "type": "thin_liquidity_ask",
                            "price_range": [float(lower_price), float(upper_price)],
                            "gap_size": float(gap_size),
                            "relative_gap": float(gap_size / np.mean(ask_gaps)),
                            "confidence": min(
                                10, float(gap_size / np.mean(ask_gaps) / 2)
                            ),
                        }
                    )

            # Sort by confidence
            stop_hunting_zones.sort(key=lambda x: x["confidence"], reverse=True)

            # Filter out low-confidence zones
            stop_hunting_zones = [z for z in stop_hunting_zones if z["confidence"] > 5]

            return {"zones": stop_hunting_zones, "count": len(stop_hunting_zones)}

        except Exception as e:
            logger.error(f"Error in detecting stop hunting zones: {e}")
            return {"zones": []}

    def _identify_spoofing_patterns(self, orderbook):
        """
        NEW: Identifies potential spoofing patterns in the orderbook
        """
        try:
            if not orderbook or not orderbook["bids"] or not orderbook["asks"]:
                return {"patterns": []}

            # Analyze historical orderbooks
            if len(self.orderbook_history) < 3:
                return {"patterns": []}

            # Extract current prices and volumes
            current_bids = {order[0]: order[1] for order in orderbook["bids"]}
            current_asks = {order[0]: order[1] for order in orderbook["asks"]}

            # Get previous orderbook
            prev_orderbook = self.orderbook_history[-2]["raw"]
            prev_bids = {order[0]: order[1] for order in prev_orderbook["bids"]}
            prev_asks = {order[0]: order[1] for order in prev_orderbook["asks"]}

            spoofing_patterns = []

            # Look for large orders that appeared and disappeared quickly
            for price, volume in prev_bids.items():
                # Large order in previous orderbook
                if volume > np.mean(list(prev_bids.values())) * 3:
                    # Order disappeared or significantly reduced
                    if price not in current_bids or current_bids[price] < volume * 0.5:
                        # Calculate price impact (if price moved down after large bid disappeared)
                        price_impact = (
                            orderbook["bids"][0][0] if orderbook["bids"] else 0
                        ) - (
                            prev_orderbook["bids"][0][0]
                            if prev_orderbook["bids"]
                            else 0
                        )

                        if (
                            price_impact < 0
                        ):  # Price moved down (expected for bid spoofing)
                            spoofing_patterns.append(
                                {
                                    "type": "bid_spoof",
                                    "price": float(price),
                                    "original_volume": float(volume),
                                    "current_volume": float(current_bids.get(price, 0)),
                                    "price_impact": float(price_impact),
                                    "confidence": min(
                                        10, abs(price_impact) * 100 / price * 5
                                    ),
                                }
                            )

            for price, volume in prev_asks.items():
                # Large order in previous orderbook
                if volume > np.mean(list(prev_asks.values())) * 3:
                    # Order disappeared or significantly reduced
                    if price not in current_asks or current_asks[price] < volume * 0.5:
                        # Calculate price impact (if price moved up after large ask disappeared)
                        price_impact = (
                            orderbook["asks"][0][0] if orderbook["asks"] else 0
                        ) - (
                            prev_orderbook["asks"][0][0]
                            if prev_orderbook["asks"]
                            else 0
                        )

                        if (
                            price_impact > 0
                        ):  # Price moved up (expected for ask spoofing)
                            spoofing_patterns.append(
                                {
                                    "type": "ask_spoof",
                                    "price": float(price),
                                    "original_volume": float(volume),
                                    "current_volume": float(current_asks.get(price, 0)),
                                    "price_impact": float(price_impact),
                                    "confidence": min(
                                        10, abs(price_impact) * 100 / price * 5
                                    ),
                                }
                            )

            # Sort by confidence
            spoofing_patterns.sort(key=lambda x: x["confidence"], reverse=True)

            return {"patterns": spoofing_patterns, "count": len(spoofing_patterns)}

        except Exception as e:
            logger.error(f"Error in identifying spoofing patterns: {e}")
            return {"patterns": []}

    def _identify_price_clusters(self, prices, volumes, eps=0.001):
        """
        Helper method to identify clusters of prices for manipulation detection
        """
        try:
            if len(prices) < 5:
                return []

            # Create feature matrix (just prices for clustering)
            X = prices.reshape(-1, 1)

            # Apply DBSCAN to identify price clusters
            clustering = DBSCAN(eps=eps, min_samples=2).fit(X)
            labels = clustering.labels_

            # Analyze clusters
            clusters = []
            for label in set(labels):
                if label == -1:  # Skip noise points
                    continue

                mask = labels == label
                cluster_prices = prices[mask]
                cluster_volumes = volumes[mask]

                if len(cluster_prices) >= 2:  # At least 2 points in cluster
                    cluster_info = {
                        "center": float(np.mean(cluster_prices)),
                        "min_price": float(np.min(cluster_prices)),
                        "max_price": float(np.max(cluster_prices)),
                        "total_volume": float(np.sum(cluster_volumes)),
                        "point_count": int(len(cluster_prices)),
                        "density": float(
                            len(cluster_prices)
                            / (
                                np.max(cluster_prices)
                                - np.min(cluster_prices)
                                + 0.00001
                            )
                        ),
                    }
                    clusters.append(cluster_info)

            return clusters

        except Exception as e:
            logger.error(f"Error in identifying price clusters: {e}")
            return []

    ##################################################################################################################################################
    ############################################################### Pattern Recognition ##############################################################
    ##################################################################################################################################################
    
    def _detect_harmonic_patterns(self, data, tolerance=0.05):
        """
        Detect harmonic patterns in price data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing OHLCV data
        tolerance : float
            Tolerance for pattern ratios
        
        Returns:
        --------
        list: List of detected patterns with details
        """
        try:
            if data is None or len(data) < 30:
                logger.warning("Insufficient data for harmonic pattern detection")
                return []
            
            detected_patterns = []
            
            # Extract price data
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Find potential pattern pivots
            pivot_highs = self._find_pivot_points(highs, 'high', window=5)
            pivot_lows = self._find_pivot_points(lows, 'low', window=5)
            
            # Combine pivots in chronological order
            pivots = []
            for idx in pivot_highs:
                pivots.append({'idx': idx, 'type': 'high', 'price': highs[idx]})
            for idx in pivot_lows:
                pivots.append({'idx': idx, 'type': 'low', 'price': lows[idx]})
            
            # Sort by index
            pivots.sort(key=lambda x: x['idx'])
            
            # Need at least 5 pivots for a harmonic pattern
            if len(pivots) < 5:
                return []
            
            # Look for pattern candidates
            for i in range(len(pivots) - 4):
                # Extract potential XABCD points
                points = pivots[i:i+5]
                
                # Ensure alternating high/low pivots
                valid_sequence = True
                for j in range(1, len(points)):
                    if points[j]['type'] == points[j-1]['type']:
                        valid_sequence = False
                        break
                
                if not valid_sequence:
                    continue
                
                # Extract price values and calculate retracement ratios
                X = points[0]['price']
                A = points[1]['price']
                B = points[2]['price']
                C = points[3]['price']
                D = points[4]['price']
                
                # XA movement
                XA = abs(A - X)
                
                # AB retracement
                AB = abs(B - A)
                AB_ratio = AB / XA
                
                # BC retracement
                BC = abs(C - B)
                BC_ratio = BC / AB
                
                # CD retracement
                CD = abs(D - C)
                CD_ratio = CD / BC
                
                # XD movement (potential profit target)
                XD = abs(D - X)
                XD_ratio = XD / XA
                
                # Butterfly Pattern: AB = 0.786, BC = 0.382-0.886, CD = 1.618-2.618, AD = 0.786
                if self._is_in_range(AB_ratio, 0.786, tolerance) and \
                self._is_in_range(BC_ratio, 0.382, 0.886, tolerance) and \
                self._is_in_range(CD_ratio, 1.618, 2.618, tolerance):
                    
                    # Verify pattern type (bullish/bearish)
                    pattern_bias = 'bullish' if X > D else 'bearish'
                    
                    detected_patterns.append({
                        'name': 'Butterfly',
                        'bias': pattern_bias,
                        'quality': 8.0,  # Quality score 0-10
                        'points': {
                            'X': {'idx': points[0]['idx'], 'price': X},
                            'A': {'idx': points[1]['idx'], 'price': A},
                            'B': {'idx': points[2]['idx'], 'price': B},
                            'C': {'idx': points[3]['idx'], 'price': C},
                            'D': {'idx': points[4]['idx'], 'price': D}
                        },
                        'ratios': {
                            'AB': AB_ratio,
                            'BC': BC_ratio,
                            'CD': CD_ratio,
                            'XD': XD_ratio
                        },
                        'target': D,
                        'stop_loss': C,  # Conservative stop loss at point C
                        'potential_reversal': True
                    })
                    
                    logger.info(f"Detected {pattern_bias} Butterfly pattern")
                
                # Gartley Pattern: AB = 0.618, BC = 0.382-0.886, CD = 1.272-1.618
                elif self._is_in_range(AB_ratio, 0.618, tolerance) and \
                    self._is_in_range(BC_ratio, 0.382, 0.886, tolerance) and \
                    self._is_in_range(CD_ratio, 1.272, 1.618, tolerance):
                    
                    pattern_bias = 'bullish' if X > D else 'bearish'
                    
                    detected_patterns.append({
                        'name': 'Gartley',
                        'bias': pattern_bias,
                        'quality': 7.5,
                        'points': {
                            'X': {'idx': points[0]['idx'], 'price': X},
                            'A': {'idx': points[1]['idx'], 'price': A},
                            'B': {'idx': points[2]['idx'], 'price': B},
                            'C': {'idx': points[3]['idx'], 'price': C},
                            'D': {'idx': points[4]['idx'], 'price': D}
                        },
                        'ratios': {
                            'AB': AB_ratio,
                            'BC': BC_ratio,
                            'CD': CD_ratio,
                            'XD': XD_ratio
                        },
                        'target': D,
                        'stop_loss': C,
                        'potential_reversal': True
                    })
                    
                    logger.info(f"Detected {pattern_bias} Gartley pattern")
                
                # Bat Pattern: AB = 0.382-0.5, BC = 0.382-0.886, CD = 1.618-2.618, AD = 0.886
                elif self._is_in_range(AB_ratio, 0.382, 0.5, tolerance) and \
                    self._is_in_range(BC_ratio, 0.382, 0.886, tolerance) and \
                    self._is_in_range(CD_ratio, 1.618, 2.618, tolerance):
                    
                    pattern_bias = 'bullish' if X > D else 'bearish'
                    
                    detected_patterns.append({
                        'name': 'Bat',
                        'bias': pattern_bias,
                        'quality': 8.0,
                        'points': {
                            'X': {'idx': points[0]['idx'], 'price': X},
                            'A': {'idx': points[1]['idx'], 'price': A},
                            'B': {'idx': points[2]['idx'], 'price': B},
                            'C': {'idx': points[3]['idx'], 'price': C},
                            'D': {'idx': points[4]['idx'], 'price': D}
                        },
                        'ratios': {
                            'AB': AB_ratio,
                            'BC': BC_ratio,
                            'CD': CD_ratio,
                            'XD': XD_ratio
                        },
                        'target': D,
                        'stop_loss': C,
                        'potential_reversal': True
                    })
                    
                    logger.info(f"Detected {pattern_bias} Bat pattern")
                
                # Crab Pattern: AB = 0.382-0.618, BC = 0.382-0.886, CD = 2.618-3.618, AD = 1.618
                elif self._is_in_range(AB_ratio, 0.382, 0.618, tolerance) and \
                    self._is_in_range(BC_ratio, 0.382, 0.886, tolerance) and \
                    self._is_in_range(CD_ratio, 2.618, 3.618, tolerance):
                    
                    pattern_bias = 'bullish' if X > D else 'bearish'
                    
                    detected_patterns.append({
                        'name': 'Crab',
                        'bias': pattern_bias,
                        'quality': 8.5,
                        'points': {
                            'X': {'idx': points[0]['idx'], 'price': X},
                            'A': {'idx': points[1]['idx'], 'price': A},
                            'B': {'idx': points[2]['idx'], 'price': B},
                            'C': {'idx': points[3]['idx'], 'price': C},
                            'D': {'idx': points[4]['idx'], 'price': D}
                        },
                        'ratios': {
                            'AB': AB_ratio,
                            'BC': BC_ratio,
                            'CD': CD_ratio,
                            'XD': XD_ratio
                        },
                        'target': D,
                        'stop_loss': C,
                        'potential_reversal': True
                    })
                    
                    logger.info(f"Detected {pattern_bias} Crab pattern")
            
            # Sort by quality
            detected_patterns.sort(key=lambda x: x['quality'], reverse=True)
            
            # Return only the most significant patterns (max 3)
            return detected_patterns[:3]
        
        except Exception as e:
            logger.error(f"Error in harmonic pattern detection: {e}")
            return []

    def _find_pivot_points(self, data, pivot_type='high', window=5):
        """
        Find pivot points in price data
        
        Parameters:
        -----------
        data : numpy.array
            Price data (high or low)
        pivot_type : str
            'high' or 'low'
        window : int
            Window size for pivot detection
        
        Returns:
        --------
        list: Indices of pivot points
        """
        pivots = []
        
        # Need at least 2*window+1 data points
        if len(data) < 2 * window + 1:
            return pivots
        
        for i in range(window, len(data) - window):
            if pivot_type == 'high':
                # Check if current point is higher than all points in window
                if data[i] == max(data[i-window:i+window+1]):
                    pivots.append(i)
            else:  # low
                # Check if current point is lower than all points in window
                if data[i] == min(data[i-window:i+window+1]):
                    pivots.append(i)
        
        return pivots

    def _is_in_range(self, value, target, upper=None, tolerance=0.05):
        """
        Check if value is within target range with tolerance
        
        Parameters:
        -----------
        value : float
            Value to check
        target : float
            Target value or lower bound
        upper : float, optional
            Upper bound (if None, only check against target)
        tolerance : float
            Tolerance percentage
        
        Returns:
        --------
        bool: True if value is within range
        """
        if upper is None:
            # Check against single target value
            return target * (1 - tolerance) <= value <= target * (1 + tolerance)
        else:
            # Check if value is within range [target, upper]
            return target * (1 - tolerance) <= value <= upper * (1 + tolerance)


    ##################################################################################################################################################
    ################################################################ Signal Generation ###############################################################
    ##################################################################################################################################################
    
    def run_liquidity_sweep_strategy(self):
        """
        Enhanced implementation of the Liquidity Sweep strategy with configurable signal weights
        """
        try:
            if self.data is None or self.orderbook_data is None:
                logger.warning("Missing data for liquidity sweep strategy")
                return None
                    
            # Get latest market data
            last_candle = self.data.iloc[-1] if len(self.data) > 0 else None
            if last_candle is None:
                return None
            
            # Check if models need retraining
            if hasattr(self, '_check_model_retraining'):
                self._check_model_retraining()
                    
            # Initialize score and signals
            sweep_score = 0
            signal_strength = 0
            buy_signals = 0
            sell_signals = 0
            direction = None
            signals_info = {}
            quality_score = 5.0  # Default neutral quality
            
            # Get signal weight multipliers (default to 1.0 if not set)
            orderbook_multiplier = self.params.get("orderbook_multiplier", 1.0)
            technical_multiplier = self.params.get("technical_multiplier", 1.0)
            
            logger.info(f"=== DETAILED MARKET ANALYSIS ===")
            logger.info(f"Current price: {last_candle['close']:.2f}")
            current_price = last_candle['close']
            
            # 1. Market Regime Analysis
            current_regime = last_candle.get('regime', 'neutral')
            logger.info(f"Current market regime: {current_regime}")
            signals_info['market_regime'] = current_regime
            
            # Apply regime-specific biases
            if current_regime == "strong_uptrend":
                buy_signals += 0.5 * technical_multiplier
                logger.info(f"BUY bias +{0.5 * technical_multiplier:.2f} from strong_uptrend regime")
            elif current_regime == "strong_downtrend":
                sell_signals += 0.5 * technical_multiplier
                logger.info(f"SELL bias +{0.5 * technical_multiplier:.2f} from strong_downtrend regime")
            
            # 2. Orderbook Analysis and liquidity pattern detection
            logger.info(f"--- Orderbook Analysis ---")
            orderbook_score, orderbook_signals = self._analyze_orderbook_for_signals()
            
            if orderbook_signals:
                signals_info['orderbook_patterns'] = orderbook_signals
                
                # Add signals from orderbook analysis with multiplier
                for signal in orderbook_signals:
                    signal_type = signal.get('type', '')
                    signal_strength = signal.get('strength', 0)
                    normalized_strength = min(1.0, signal_strength / 10.0) * orderbook_multiplier
                    
                    if 'buy' in signal_type.lower() or 'bull' in signal_type.lower() or 'support' in signal_type.lower():
                        buy_signals += normalized_strength
                        logger.info(f"BUY signal +{normalized_strength:.2f}: {signal_type}")
                    elif 'sell' in signal_type.lower() or 'bear' in signal_type.lower() or 'resistance' in signal_type.lower():
                        sell_signals += normalized_strength
                        logger.info(f"SELL signal +{normalized_strength:.2f}: {signal_type}")
            
            # 3. Technical Analysis from OHLCV data
            logger.info(f"--- Technical Analysis ---")
            
            # RSI analysis with technical multiplier
            rsi = last_candle.get('rsi', 50)
            logger.info(f"RSI: {rsi:.2f}")
            if rsi < 30:
                rsi_signal = 1.0 * technical_multiplier
                buy_signals += rsi_signal
                signals_info['rsi'] = f"Oversold ({rsi:.2f})"
                logger.info(f"BUY signal +{rsi_signal:.2f}: RSI oversold ({rsi:.2f})")
            elif rsi > 70:
                rsi_signal = 1.0 * technical_multiplier
                sell_signals += rsi_signal
                signals_info['rsi'] = f"Overbought ({rsi:.2f})"
                logger.info(f"SELL signal +{rsi_signal:.2f}: RSI overbought ({rsi:.2f})")
            
            # Trend analysis with technical multiplier
            ema50 = last_candle.get('ema50', last_candle['close'])
            ema200 = last_candle.get('ema200', last_candle['close'])
            
            logger.info(f"EMA50: {ema50:.2f}, EMA200: {ema200:.2f}")
            if ema50 > ema200:
                trend_signal = 0.5 * technical_multiplier
                buy_signals += trend_signal
                signals_info['trend'] = "Uptrend (EMA50 > EMA200)"
                logger.info(f"BUY signal +{trend_signal:.2f}: Uptrend (EMA50 > EMA200)")
            elif ema50 < ema200:
                trend_signal = 0.5 * technical_multiplier
                sell_signals += trend_signal
                signals_info['trend'] = "Downtrend (EMA50 < EMA200)"
                logger.info(f"SELL signal +{trend_signal:.2f}: Downtrend (EMA50 < EMA200)")
            
            # Check EMA slopes for momentum
            if 'ema50_slope' in last_candle:
                ema_slope = last_candle['ema50_slope']
                signals_info['ema_slope'] = f"{ema_slope:.5f}"
                if ema_slope > 0.001:  # Upward momentum
                    slope_signal = 0.3 * technical_multiplier
                    buy_signals += slope_signal
                    logger.info(f"BUY signal +{slope_signal:.2f}: Positive EMA slope ({ema_slope:.5f})")
                elif ema_slope < -0.001:  # Downward momentum
                    slope_signal = 0.3 * technical_multiplier
                    sell_signals += slope_signal
                    logger.info(f"SELL signal +{slope_signal:.2f}: Negative EMA slope ({ema_slope:.5f})")
            
            # Volume analysis with technical multiplier
            vol_sma20 = last_candle.get('volume_sma20', last_candle['volume'])
            vol_ratio = last_candle['volume'] / vol_sma20 if vol_sma20 > 0 else 1.0
            logger.info(f"Volume: {last_candle['volume']:.2f}, Average: {vol_sma20:.2f}, Ratio: {vol_ratio:.2f}x")
            
            if last_candle['volume'] > vol_sma20 * 1.5:
                signals_info['volume'] = f"High volume ({vol_ratio:.2f}x above average)"
                logger.info(f"High volume detected: {vol_ratio:.2f}x above average")
                
                # Determine direction bias based on price action with high volume
                if last_candle['close'] > last_candle['open']:
                    vol_signal = 0.3 * technical_multiplier
                    buy_signals += vol_signal
                    logger.info(f"BUY signal +{vol_signal:.2f}: High volume with price increase")
                elif last_candle['close'] < last_candle['open']:
                    vol_signal = 0.3 * technical_multiplier
                    sell_signals += vol_signal
                    logger.info(f"SELL signal +{vol_signal:.2f}: High volume with price decrease")
            
            # Momentum indicators with technical multiplier
            macd = last_candle.get('macd', 0)
            macd_signal = last_candle.get('macd_signal', 0)
            macd_hist = last_candle.get('macd_hist', 0)
            logger.info(f"MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Hist: {macd_hist:.4f}")
            
            if macd > macd_signal:
                macd_sig = 0.4 * technical_multiplier
                buy_signals += macd_sig
                logger.info(f"BUY signal +{macd_sig:.2f}: MACD > Signal")
                if macd_hist > 0 and macd_hist > last_candle.get('macd_hist_prev', 0):
                    macd_hist_sig = 0.2 * technical_multiplier
                    buy_signals += macd_hist_sig
                    logger.info(f"BUY signal +{macd_hist_sig:.2f}: MACD histogram increasing")
            elif macd < macd_signal:
                macd_sig = 0.4 * technical_multiplier
                sell_signals += macd_sig
                logger.info(f"SELL signal +{macd_sig:.2f}: MACD < Signal")
                if macd_hist < 0 and macd_hist < last_candle.get('macd_hist_prev', 0):
                    macd_hist_sig = 0.2 * technical_multiplier
                    sell_signals += macd_hist_sig
                    logger.info(f"SELL signal +{macd_hist_sig:.2f}: MACD histogram decreasing")
            
            # Mean Reversion Check
            if self.params.get("mean_reversion_filter", True):
                try:
                    # Check for mean reversion opportunity
                    if 'zscore' in last_candle:
                        zscore = last_candle['zscore']
                        logger.info(f"Price z-score: {zscore:.2f}")
                        
                        deviation_threshold = self.params.get("deviation_threshold", 2.0)
                        
                        if zscore > deviation_threshold:  # Significantly above mean
                            mr_signal = 0.5 * technical_multiplier
                            sell_signals += mr_signal
                            logger.info(f"SELL signal +{mr_signal:.2f}: Price significantly above mean (z-score: {zscore:.2f})")
                            signals_info['mean_reversion'] = f"Price above mean (z: {zscore:.2f})"
                        elif zscore < -deviation_threshold:  # Significantly below mean
                            mr_signal = 0.5 * technical_multiplier
                            buy_signals += mr_signal
                            logger.info(f"BUY signal +{mr_signal:.2f}: Price significantly below mean (z-score: {zscore:.2f})")
                            signals_info['mean_reversion'] = f"Price below mean (z: {zscore:.2f})"
                except Exception as e:
                    logger.error(f"Error in mean reversion check: {e}")
            
            # Determine overall direction
            logger.info(f"--- Signal Summary ---")
            logger.info(f"BUY Signals: {buy_signals:.2f}")
            logger.info(f"SELL Signals: {sell_signals:.2f}")
            
            # Check for active position first
            if self.active_trade is not None:
                logger.info(f"Active position exists ({self.active_trade['side']}). No new signals generated.")
                return None
            
            # Calculate signal strength and determine direction
            if buy_signals > sell_signals:
                direction = 'buy'
                signal_strength = buy_signals - sell_signals
                logger.info(f"Prevailing direction: BUY (strength: {signal_strength:.2f})")
            elif sell_signals > buy_signals:
                direction = 'sell'
                signal_strength = sell_signals - buy_signals
                logger.info(f"Prevailing direction: SELL (strength: {signal_strength:.2f})")
            else:
                logger.info(f"No prevailing direction")
                
            # Apply ML-based quality score prediction if available
            try:
                # Gather market features for quality prediction
                market_features = {
                    'rsi': last_candle.get('rsi', 50),
                    'adx': last_candle.get('adx', 15),
                    'atr_percent': last_candle.get('atr_percent', 1.0),
                    'volume_ratio': vol_ratio,
                    'macd': last_candle.get('macd', 0),
                    'close': last_candle['close'],
                    'regime': current_regime
                }
                
                # Predict quality score using ML model if available
                if hasattr(self, '_predict_signal_quality'):
                    quality_score = self._predict_signal_quality(market_features)
                    logger.info(f"ML-predicted signal quality: {quality_score:.2f}/10")
                    
                    # Adjust signal strength based on quality score
                    if direction is not None:
                        # Quality score influence
                        quality_weight = self.params.get("ml_prediction_weight", 0.3)
                        adjusted_strength = signal_strength * (1 + (quality_score - 5) / 10 * quality_weight)
                        
                        logger.info(f"Signal strength adjusted by quality score: {signal_strength:.2f} -> {adjusted_strength:.2f}")
                        signal_strength = adjusted_strength
            except Exception as e:
                logger.error(f"Error in quality prediction: {e}")
            
            # Apply session-based adjustments if enabled
            if self.params.get("trade_session_filter", True):
                try:
                    # Determine current trading session based on UTC time
                    current_time = datetime.now().hour
                    
                    # Simple classification of trading sessions
                    if 0 <= current_time < 8:
                        current_session = "asia"
                        session_weight = self.params.get("session_filter_weights", {}).get("asia", 0.8)
                    elif 8 <= current_time < 16:
                        current_session = "europe"
                        session_weight = self.params.get("session_filter_weights", {}).get("europe", 1.0)
                    else:
                        current_session = "us"
                        session_weight = self.params.get("session_filter_weights", {}).get("us", 0.9)
                    
                    # Apply session weight
                    original_strength = signal_strength
                    signal_strength *= session_weight
                    
                    logger.info(f"Current session: {current_session.upper()}, weight: {session_weight}")
                    logger.info(f"Signal strength adjusted by session: {original_strength:.2f} -> {signal_strength:.2f}")
                    
                    signals_info['trading_session'] = f"{current_session} (weight: {session_weight})"
                except Exception as e:
                    logger.error(f"Error in session filtering: {e}")
            
            # Calculate final sweep score
            sweep_score = signal_strength * 2  # Scale up for better granularity
            
            # Apply quality score threshold if applicable
            if self.params.get("use_gradient_boosting", True):
                quality_threshold = self.params.get("quality_threshold", 0.5) * 10  # Convert to 0-10 scale
                
                if quality_score < quality_threshold:
                    logger.info(f"Signal rejected due to low quality score: {quality_score:.2f} < threshold {quality_threshold:.2f}")
                    return None
            
            # Final decision and score
            logger.info(f"Final sweep score: {sweep_score:.2f}")
            
            # Minimum threshold for signal generation 
            min_score_threshold = self.params.get("min_signal_score", 25.0) / 10  # Convert to our scale
            
            if direction is None or sweep_score < min_score_threshold:
                logger.info(f"Sweep score too low ({sweep_score:.2f}) or no direction. No signal generated.")
                return None
                    
            # Calculate stop loss and take profit based on ATR or fixed percentage
            # Get base SL and TP percentages (these would be set when running the bot)
            base_sl_percent = self.params.get('base_sl_percent', 1.0)  # Default 1%
            base_tp_percent = self.params.get('base_tp_percent', 1.0)  # Default 1%
            
            # Apply regime-specific multipliers
            sl_factor = self.params.get('sl_factor', 0.5)
            tp_factor = self.params.get('tp_factor', 1.0)
            
            # Adjust based on quality score if enabled
            quality_score_adjustment = self.params.get('quality_score_adjustment', False)
            if quality_score_adjustment and quality_score is not None:
                # For high quality scores, tighten stop loss
                if quality_score >= 8.0:
                    sl_factor *= 0.5  # Tighter stop (e.g., 0.5 * 0.3 = 0.15% for ranging)
            
            # Calculate stop loss and take profit based on ATR or fixed percentage
            if 'atr' in self.data.columns and self.params.get('use_atr_for_stops', True):
                atr = last_candle['atr']
                
                # Convert percentage-based SL/TP to ATR multipliers
                # This assumes 1 ATR approximately equals base_sl_percent of price
                atr_to_percent_ratio = current_price * (base_sl_percent/100) / atr
                
                # Apply regime and quality factor to ATR multiplier
                sl_atr_multiplier = sl_factor * atr_to_percent_ratio
                tp_atr_multiplier = tp_factor * atr_to_percent_ratio
                
                if direction == 'buy':
                    stop_loss = current_price - (atr * sl_atr_multiplier)
                    take_profit = current_price + (atr * tp_atr_multiplier)
                else:  # sell
                    stop_loss = current_price + (atr * sl_atr_multiplier)
                    take_profit = current_price - (atr * tp_atr_multiplier)
                    
                logger.info(f"ATR-based stops: ATR={atr:.4f}, Adjusted SL={sl_factor:.2f}x, TP={tp_factor:.2f}x")
                logger.info(f"Base SL%={base_sl_percent:.2f}%, Adjusted SL%={(sl_factor * base_sl_percent):.2f}%")
                logger.info(f"Base TP%={base_tp_percent:.2f}%, Adjusted TP%={(tp_factor * base_tp_percent):.2f}%")
            else:
                # Use fixed percentage with regime multipliers
                sl_percent = base_sl_percent * sl_factor  # Apply regime-specific multiplier
                tp_percent = base_tp_percent * tp_factor  # Apply regime-specific multiplier
                
                if direction == 'buy':
                    stop_loss = current_price * (1 - sl_percent/100)
                    take_profit = current_price * (1 + tp_percent/100)
                else:  # sell
                    stop_loss = current_price * (1 + sl_percent/100)
                    take_profit = current_price * (1 - tp_percent/100)
                    
                logger.info(f"Percentage-based stops: SL={sl_percent:.2f}%, TP={tp_percent:.2f}%")
                logger.info(f"Base SL={base_sl_percent:.2f}%, Factor={sl_factor:.2f}x")
                logger.info(f"Base TP={base_tp_percent:.2f}%, Factor={tp_factor:.2f}x")
            
            current_price = None
            if self.orderbook_data and "metrics" in self.orderbook_data:
                metrics = self.orderbook_data["metrics"]
                bid = metrics.get("best_bid", 0)
                ask = metrics.get("best_ask", 0)
                if bid > 0 and ask > 0:
                    # Use mid price as current price
                    current_price = (bid + ask) / 2
                    logger.info(f"Using current mid price: {current_price}")

            # Fallback to last candle close if unable to get current price
            if not current_price:
                current_price = last_candle['close']
                logger.info(f"Using last candle close as price: {current_price}")
            
            # Prepare the final signal
            signal = {
                'side': direction,
                'sweep_score': sweep_score,
                'price': current_price,
                'timestamp': datetime.now(),
                'quality_score': quality_score,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_factor' : sl_factor,
                'tp_factor' : tp_factor,
                'details': signals_info
            }
            
            logger.info(f"!!! SIGNAL GENERATED !!! {direction.upper()} with score {sweep_score:.2f}")
            return signal
                
        except Exception as e:
            logger.error(f"Error in liquidity sweep strategy: {e}")
            return None

    ##################################################################################################################################################
    ############################################################## Performance Tracking ##############################################################
    ##################################################################################################################################################

    def _load_trade_history(self):
        """Load trade history from file if available"""
        try:
            if os.path.exists("trade_history.json"):
                with open("trade_history.json", "r") as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
                self._update_performance_metrics()
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            self.trade_history = []

    def _update_performance_metrics(self):
        """Update performance metrics based on trade history"""
        try:
            if not self.trade_history:
                return

            # Calculate basic metrics
            wins = [t for t in self.trade_history if t.get("profit_pct", 0) > 0]
            losses = [t for t in self.trade_history if t.get("profit_pct", 0) <= 0]

            if len(self.trade_history) > 0:
                self.performance_metrics["win_rate"] = len(wins) / len(
                    self.trade_history
                )

            if wins:
                self.performance_metrics["avg_win"] = sum(
                    t.get("profit_pct", 0) for t in wins
                ) / len(wins)

            if losses:
                self.performance_metrics["avg_loss"] = sum(
                    t.get("profit_pct", 0) for t in losses
                ) / len(losses)

            # Calculate profit factor
            total_wins = sum(t.get("profit_pct", 0) for t in wins) if wins else 0
            total_losses = (
                abs(sum(t.get("profit_pct", 0) for t in losses)) if losses else 0
            )

            if total_losses > 0:
                self.performance_metrics["profit_factor"] = total_wins / total_losses
            else:
                self.performance_metrics["profit_factor"] = (
                    float("inf") if total_wins > 0 else 0
                )

            # Calculate maximum consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0

            for trade in self.trade_history:
                if trade.get("profit_pct", 0) <= 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(
                        max_consecutive_losses, consecutive_losses
                    )
                else:
                    consecutive_losses = 0

            self.performance_metrics["consecutive_losses"] = consecutive_losses
            self.performance_metrics["max_consecutive_losses"] = max_consecutive_losses

            # Calculate maximum drawdown
            balance_curve = []
            cumulative_return = 0

            for trade in self.trade_history:
                pnl = trade.get("profit_pct", 0) / 100 * trade.get("size_usd", 0)
                cumulative_return += pnl
                balance_curve.append(cumulative_return)

            if balance_curve:
                peak = max(balance_curve)
                idx_peak = balance_curve.index(peak)

                # Find minimum after peak
                if idx_peak < len(balance_curve) - 1:
                    trough = min(balance_curve[idx_peak:])
                    drawdown = (peak - trough) / peak if peak > 0 else 0
                    self.performance_metrics["drawdown"] = drawdown

            logger.info(
                f"Performance metrics updated: Win rate: {self.performance_metrics['win_rate']:.2f}, "
                + f"Profit factor: {self.performance_metrics['profit_factor']:.2f}, "
                + f"Max drawdown: {self.performance_metrics['drawdown']:.2f}"
            )

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    ##################################################################################################################################################
    ############################################################# Monitoring and Testing #############################################################
    ##################################################################################################################################################
