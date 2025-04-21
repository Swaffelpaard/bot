import os
from dotenv import load_dotenv
import time
import logging
from datetime import datetime

####### CHANGED TP/SL TO % FOR TESTING #######


# Import both bot classes
from ExecutionBotV7 import ExecutionBot
from SignalBotV7 import SignalBot

# Load credentials
load_dotenv()
wallet_address = os.environ.get("wallet")
private_key = os.environ.get("key")

# Check if credentials are loaded
if not wallet_address or not private_key:
    print("Error: Missing Hyperliquid credentials. Check your .env file.")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("TradingBot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("HyperComboBot")

def main():
    print(f"\n=== HyperComboBot Starting ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbol: ETH/USDC:USDC")
    print(f"Mode: Live Trading\n")
    
    base_sl_percent = 1.0
    base_tp_percent = 1.0

    # Risk management parameters
    risk_params = {
        # Position sizing
        'risk_per_trade': 0.01,       # Risk 1% of account per trade
        'leverage': 2,                # x leverage
        
        # Stop-loss settings
        'use_signal_sl_tp': True,      # Use signal bot for SL/TP, False for ATR-based stops or fixed percentage as entered below
        'use_atr_for_stops': False,    # Use ATR for dynamic stops, False for fixed percentage
        'stop_loss_atr_multiple': 1.25,  # Stop loss at x * ATR
        'take_profit_atr_multiple': 1.5, # Take profit at x * ATR
        
        # Fixed percentage stops (backup if ATR not available)
        'stop_loss_percent': 0.65,     # Stop loss at x % from entry
        'take_profit_percent': 0.5,   # Take profit at x % from entry
        
        # NEW: Trailing stop configuration
        'use_trailing_stop': True,     # Toggle trailing stop mechanism

        # Position management
        'min_holding_time_minutes': 30    # Minimum time to hold position
    }
    
    try:
        # 1. Initialize execution bot first (handles the exchange setup)
        execution_bot = ExecutionBot(
            symbol='ETH/USDC:USDC', 
            timeframe='5m',
            leverage=risk_params['leverage'],
            margin_mode="isolated",
            risk_per_trade=risk_params['risk_per_trade'],
            stop_loss_atr_multiple=risk_params['stop_loss_atr_multiple'],
            take_profit_atr_multiple=risk_params['take_profit_atr_multiple'],
            stop_loss_percent=risk_params['stop_loss_percent'],
            take_profit_percent=risk_params['take_profit_percent'],
            use_atr_for_stops=risk_params['use_atr_for_stops'],
            min_holding_time_minutes=risk_params['min_holding_time_minutes'],
            wallet_address=wallet_address,
            private_key=private_key,
            use_signal_sl_tp=risk_params['use_signal_sl_tp'],  # Use signal bot for SL/TP
            use_trailing_stop=risk_params['use_trailing_stop'],  # Add this parameter
            debug_mode=True,
            base_sl_percent=base_sl_percent,  # Base stop loss percentage
            base_tp_percent=base_tp_percent,  # Base take profit percentage
            monitor_interval=10   # 5-second interval for position monitoring
        )
        
        logger.info("ExecutionBot initialized successfully")
        
        # 2. Initialize signal bot, passing the exchange from execution bot
        signal_bot = SignalBot(
            symbol='ETH/USDC:USDC', 
            timeframe='5m',
            wallet_address=wallet_address,
            private_key=private_key,
            debug_mode=True,
            base_sl_percent=base_sl_percent,  # Base stop loss percentage
            base_tp_percent=base_tp_percent,  # Base take profit percentage
            # Use exchange from execution bot - avoid creating a new connection
            # Note: This parameter would need to be added to SignalBot's __init__
            exchange=execution_bot.exchange  
        )
        
        logger.info("SignalBot initialized successfully")
        
        # 3. Connect signal bot to execution bot
        execution_bot.signal_bot = signal_bot
        
        # 4. Adjust signal weights based on your preferred strategy
        signal_bot.adjust_signal_weights({
            "orderbook_multiplier": 1.2,     # Increased orderbook influence
            "technical_multiplier": 0.8,     # Technical indicator influence
            "l2_weight": 0.7,                # Standard L2 orderbook weight
            "l3_weight": 0.6,                # Standard L3 analysis weight
            "order_flow_weight": 0.6,        # Standard order flow weight
            "orderbook_pattern_weight": 0.5, # Standard orderbook pattern weight
            "trend_weight": 0.8,             # Standard trend weight
            "quality_threshold": 0.6,        # Quality threshold (0-1)
            "minimum_signal_score": 40.0,    # Minimum signal strength to consider
        })
        
        logger.info("Signal weights adjusted")
        
        # Display account information
        try:
            balance = execution_bot.exchange.fetch_balance()
            usdc_balance = balance['total']['USDC']
            logger.info(f"Account balance: {usdc_balance} USDC")
            print(f"Trading with {usdc_balance} USDC")
            print(f"Leverage: {risk_params['leverage']}x")
            
            if risk_params['use_atr_for_stops']:
                print(f"Using ATR-based stops: SL={risk_params['stop_loss_atr_multiple']}x ATR, TP={risk_params['take_profit_atr_multiple']}x ATR")
            elif risk_params['use_signal_sl_tp']:
                print(f"Using signal-based stop multipliers based on regime parameters: Base SL={base_sl_percent}%, Base TP={base_tp_percent}%")
            else:
                print(f"Using percentage-based stops: SL={risk_params['stop_loss_percent']}%, TP={risk_params['take_profit_percent']}%")
                
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
        
        # Start the trading loop with the execution bot
        print(f"\n=== Trading Session Started ===\n")
        execution_bot.start_trading_loop(interval=60)  # 60-second trading cycle
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
