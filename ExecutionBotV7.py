import ccxt
import logging
from datetime import datetime, timedelta
import time
import random
from functools import wraps
import warnings
import threading
import os
from dotenv import load_dotenv
import websocket
import json

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
wallet_address = os.environ.get("wallet")
private_key = os.environ.get("key")

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
        logging.FileHandler("TradingBot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ExecutionBot")

# Error handling helper function
def log_error(error, context):
    """Helper function for standardized error logging"""
    error_msg = str(error).lower()
    error_type = 'unknown'
    
    # Classify the error
    for err_type, keywords in {
        'connectivity': ['timeout', 'connection', 'network', 'socket'],
        'rate_limit': ['rate limit', 'ratelimit', 'too many requests', '429'],
        'insufficient_funds': ['insufficient', 'not enough', 'balance', 'funds'],
        'order_size': ['size too small', 'amount too small', 'minimum size', 'minimum amount'],
        'price': ['price', 'invalid price', 'tick size'],
        'authentication': ['auth', 'login', 'credentials', 'signature', 'key'],
        'permission': ['permission', 'not allowed', 'forbidden'],
    }.items():
        if any(keyword in error_msg for keyword in keywords):
            error_type = err_type
            break
    
    logger.error(f"Error in {context} ({error_type}): {error}")
    return error_type



class HyperliquidWebSocket:
    def __init__(self, wallet_address):
        self.wallet_address = wallet_address
        self.ws = None
        self.position_data = None
        self.order_data = None
        self.last_update_time = 0
        self.connected = False
        
    def connect(self):
        """Connect to Hyperliquid WebSocket"""
        # Hyperliquid's WebSocket URL
        websocket_url = "wss://api.hyperliquid.xyz/ws"
        
        self.ws = websocket.WebSocketApp(
            websocket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run websocket in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for connection
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.connected:
            raise Exception("Failed to connect to WebSocket")
        
        logger.info("WebSocket connected successfully")
        
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        self.connected = True
        
        # Subscribe to user data streams
        subscription_messages = [
            {
                "method": "subscribe",
                "subscription": {
                    "type": "userEvents",
                    "user": self.wallet_address
                }
            },
            {
                "method": "subscribe",
                "subscription": {
                    "type": "userFills",
                    "user": self.wallet_address
                }
            },
            {
                "method": "subscribe",
                "subscription": {
                    "type": "userFundings",
                    "user": self.wallet_address
                }
            },
            {
                "method": "subscribe",
                "subscription": {
                    "type": "userNonFundingLedgerUpdates",
                    "user": self.wallet_address
                }
            }
        ]
        
        for msg in subscription_messages:
            self.ws.send(json.dumps(msg))
        
        logger.info("Subscribed to user data streams")
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get("channel") == "userEvents":
                # Update position data from real-time events
                if "data" in data and "positions" in data["data"]:
                    self.position_data = data["data"]["positions"]
                    self.last_update_time = time.time()
                    
                if "data" in data and "orders" in data["data"]:
                    self.order_data = data["data"]["orders"]
                    self.last_update_time = time.time()
                    
            elif data.get("channel") == "userFills":
                # Handle fill events
                logger.info(f"Fill event received: {data}")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
        
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        self.connected = False
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
    def get_position_data(self):
        """Get latest position data from WebSocket"""
        return self.position_data
    
    def get_order_data(self):
        """Get latest order data from WebSocket"""
        return self.order_data

class ExecutionBot:

    ##################################################################################################################################################
    ############################################################### Initialization ###################################################################
    ##################################################################################################################################################

    def __init__(self, signal_bot=None, symbol='ETH/USDC:USDC', timeframe='5m', lookback_period=10000, 
             fast_period=5, slow_period=34, signal_period=7, xtl_period=35,
             xtl_threshold=37, risk_per_trade=0.23, max_open_trades=1,
             stop_loss_atr_multiple=1.0, take_profit_atr_multiple=2.0,
             stop_loss_percent=1.0, take_profit_percent=2.0, use_trailing_stop=True, base_sl_percent=1.0, base_tp_percent=1.0, 
             use_atr_for_stops=True, use_signal_sl_tp=True, leverage=10, margin_mode="isolated", wallet_address=None, private_key=None,
             debug_mode=True, min_holding_time_minutes=30, monitor_interval=15):
        
        """
        Initialize the EWO strategy for Hyperliquid Exchange with debug options.
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (default: 'ETH/USDC:USDC')
        timeframe : str
            Candle timeframe (default: '5m')
        wallet_address : str
            Wallet address for Hyperliquid
        private_key : str
            Private key for Hyperliquid
        leverage : int
            Financial leverage to use (default: 5)
        margin_mode : str
            Margin mode ('cross' or 'isolated')
        debug_mode : bool
            Activates debug mode with more logging
        min_holding_time_minutes : int
            Minimum time to hold a position before allowing exit signals
        """
        self.monitor_interval = monitor_interval
        self.monitor_thread = None
        self.monitor_active = False
        
        self.signal_bot = signal_bot
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_period = lookback_period
        
        # Add a flag to track position opening state
        self.position_opening_in_progress = False
        self.position_opened_time = None
        self.position_verified = False
        self.min_holding_time_minutes = min_holding_time_minutes

        # EWO Parameters
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # XTL Parameters
        self.xtl_period = xtl_period
        self.xtl_threshold = xtl_threshold
        
        # Risk management parameters
        self.use_signal_sl_tp = use_signal_sl_tp
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.take_profit_atr_multiple = take_profit_atr_multiple
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.use_atr_for_stops = use_atr_for_stops
        self.min_holding_time_minutes = min_holding_time_minutes
        self.base_sl_percent = base_sl_percent
        self.base_tp_percent = base_tp_percent
        # Store trailing stop setting
        self.use_trailing_stop = use_trailing_stop

        # Trading parameters
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.debug_mode = debug_mode
        
        # Initialize Hyperliquid exchange
        self.exchange = ccxt.hyperliquid({
            "walletAddress": wallet_address,
            "privateKey": private_key,
            "options": {
                "defaultMarketSlippagePercentage": 5.0  # 5% default slippage
            }
        })

        # Initialize WebSocket connection
        self.ws_client = HyperliquidWebSocket(self.wallet_address)
        self.ws_client.connect()
                 
        # Initialize data storage
        self.data = None
        self.positions = []
        self.trade_history = []
        self.open_orders = []
        self.trailing_stop_data = None
        self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
        
        # Rate limiting controls
        self.last_api_call_time = {}  # Track last call time per endpoint
        self.current_backoff = 1.0  # Current backoff multiplier
        self.rate_limited = False   # Flag for when we're being rate limited
        self.position_cache = None  # Store position data
        self.position_cache_time = 0  # When position was last retrieved
        self.price_cache = None     # Store price data
        self.price_cache_time = 0   # When price was last retrieved
        
        # Monitoring controls
        self.full_check_interval = 20  # Time between full checks (seconds)
        self.last_full_check_time = 0  # When we last did a full check

        # Initial configuration
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Configure the exchange and initial settings"""
        try:
            self.exchange.load_markets()
            logger.info(f"Markets loaded successfully. {self.symbol} available: {self.symbol in self.exchange.markets}")
            
            # Safely set leverage - will handle errors if positions exist
            leverage_set = self._safe_set_leverage(self.leverage, self.symbol)
            
            # Only attempt to set margin mode if leverage was set successfully
            if leverage_set:
                try:
                    if self.margin_mode == "isolated":
                        self.exchange.set_margin_mode("isolated", self.symbol, params={"leverage": self.leverage})
                    else:
                        self.exchange.set_leverage(self.leverage, self.symbol, params={"marginMode": "cross"})
                    
                    logger.info(f"Margin mode set to {self.margin_mode}")
                except Exception as e:
                    logger.warning(f"Could not set margin mode: {e}")

            # Verify settings and check for positions
            positions = self.get_open_positions(verbose=False)
            if positions and len(positions) > 0 and any(float(pos.get('contracts', 0)) != 0 for pos in positions):
                for position in positions:
                    actual_leverage = position.get('leverage', 'Unknown')
                    position_side = position.get('side', 'Unknown')
                    position_size = float(position.get('contracts', 0))
                    logger.info(f"Found existing {position_side} position of {position_size} contracts with {actual_leverage}x leverage")

            logger.info(f"Exchange initialized successfully")

        except Exception as e:
            log_error(e, "initializing the exchange")

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

    def _throttled_api_call(self, endpoint_name, call_function, *args, **kwargs):
        """
        Execute API calls with rate limiting controls
        
        Parameters:
        -----------
        endpoint_name : str
            Name to track this endpoint for rate limiting
        call_function : callable
            The actual API call function to execute
        *args, **kwargs : 
            Arguments to pass to the call function
        
        Returns:
        --------
        API call result or None if failed
        """
        current_time = time.time()
        min_interval = 1.0  # Minimum seconds between calls to same endpoint
        
        # Check if we need to wait
        if endpoint_name in self.last_api_call_time:
            time_since_last = current_time - self.last_api_call_time[endpoint_name]
            if time_since_last < min_interval * self.current_backoff:
                # Add jitter to prevent synchronized calls
                sleep_time = (min_interval * self.current_backoff - time_since_last) + random.uniform(0, 0.5)
                logger.debug(f"Throttling {endpoint_name} API call, waiting {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Update last call time
        self.last_api_call_time[endpoint_name] = time.time()
        
        # Execute the call with retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = call_function(*args, **kwargs)
                
                # Success - gradually reduce backoff
                if self.current_backoff > 1.0:
                    self.current_backoff = max(1.0, self.current_backoff / 1.2)
                    self.rate_limited = False
                    
                return result
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    self.rate_limited = True
                    
                    # Increase backoff factor
                    self.current_backoff *= 1.5
                    
                    if attempt < max_retries:
                        backoff_time = min_interval * self.current_backoff
                        logger.warning(f"Rate limited on {endpoint_name}. Backing off for {backoff_time:.2f}s. Retry {attempt+1}/{max_retries}")
                        time.sleep(backoff_time)
                    else:
                        logger.error(f"Max retries exceeded for {endpoint_name}")
                        return None
                else:
                    # Not a rate limit error - just log and return None
                    logger.error(f"Error in {endpoint_name}: {e}")
                    return None
                    
        return None  # Should never reach here but just in case

    ##################################################################################################################################################
    ################################################################ Order Execution #################################################################
    ##################################################################################################################################################

    def calculate_stop_and_target(self, side, entry_price):
        """
        Calculate stop loss and take profit levels based on configured settings
        
        Parameters:
        -----------
        side : str
            Position side ('buy' for long, 'sell' for short)
        entry_price : float
            Entry price of the position
        
        Returns:
        --------
        tuple: (stop_loss_price, take_profit_price)
        """
        try:
            if self.use_atr_for_stops:
                # Use ATR for dynamic stop calculation
                atr_value = self.data['atr'].iloc[-1] if 'atr' in self.data.columns else entry_price * 0.01
                
                if side == 'buy':  # Long position
                    stop_loss = entry_price - (atr_value * self.stop_loss_atr_multiple)
                    take_profit = entry_price + (atr_value * self.take_profit_atr_multiple)
                else:  # Short position
                    stop_loss = entry_price + (atr_value * self.stop_loss_atr_multiple)
                    take_profit = entry_price - (atr_value * self.take_profit_atr_multiple)
                    
                logger.info(f"ATR-based stops - ATR: {atr_value:.2f}, SL Multiple: {self.stop_loss_atr_multiple}, TP Multiple: {self.take_profit_atr_multiple}")
                
            else:
                # Use percentage-based stops
                if side == 'buy':  # Long position
                    stop_loss = entry_price * (1 - self.stop_loss_percent/100)
                    take_profit = entry_price * (1 + self.take_profit_percent/100)
                else:  # Short position
                    stop_loss = entry_price * (1 + self.stop_loss_percent/100)
                    take_profit = entry_price * (1 - self.take_profit_percent/100)
                    
                logger.info(f"Percentage-based stops - SL: {self.stop_loss_percent}%, TP: {self.take_profit_percent}%")
            
            # Round to appropriate precision (adjust as needed)
            stop_loss = round(stop_loss, 2)
            take_profit = round(take_profit, 2)
            
            return stop_loss, take_profit
            
        except Exception as e:
            log_error(e, "calculating stop and target levels")
            
            # Fallback to basic percentage stops in case of error
            if side == 'buy':
                return entry_price * 0.9935, entry_price * 1.005  # 1% SL, 2% TP
            else:
                return entry_price * 1.0065, entry_price * 0.995  # 1% SL, 2% TP

    def calculate_position_size(self, entry_price, stop_loss):
        """
        Calculate position size to use a larger percentage of available balance
        
        Parameters:
        -----------
        entry_price : float
            Entry price
        stop_loss : float
            Stop loss price
        
        Returns:
        --------
        float: Position size in contracts
        """
        try:
            # Get current balance
            balance = self.exchange.fetch_balance()
            collateral = float(balance['total']['USDC'])
            # Ensure stop_loss has a value
            if stop_loss is None:
                # Fallback: calculate stop loss using ATR or a percentage of entry price
                atr = self.data['atr'].iloc[-1] if len(self.data) > 0 else entry_price * 0.02
                stop_loss = entry_price - (atr * self.stop_loss_atr_multiple)            

            # Use more of the available collateral (98% instead of 95%)
            available_collateral = collateral * 0.98
            
            # Calculate position value (use 98% of available funds)
            target_equity_percentage = 0.98  # Use 98% of equity
            position_value = available_collateral * target_equity_percentage
            
            # Calculate leverage-adjusted position size
            position_size = position_value / entry_price * self.leverage
            
            # Implement risk-based position sizing as a minimum backstop
            # (This ensures we don't take positions that are too risky)
            risk_percentage = abs(entry_price - stop_loss) / entry_price
            max_risk_amount = collateral * 0.02  # Risk at most 2% of account on a single trade
            risk_based_position_size = max_risk_amount / risk_percentage / entry_price * self.leverage
            
            # Choose the smaller of the two position sizes (for safety)
            position_size = min(position_size, risk_based_position_size)
            
            # Round to exchange precision
            position_size = round(position_size, 3)  # Adjust based on required precision
            
            # Set minimum
            min_size = 0.001  # Minimum contract size
            position_size = max(min_size, position_size)
            
            logger.info(f"Position size: {position_size:.4f} contracts (Risk-based: {risk_based_position_size:.4f})")
            
            return position_size
        except Exception as e:
            log_error(e, "position size calculation")
            return 0.001

    def get_open_positions(self, verbose=True):
        """Retrieve open positions with enhanced reporting"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            
            # Only process if positions exist with non-zero size
            active_positions = []
            for pos in positions:
                try:
                    if float(pos.get('contracts', 0) or 0) != 0:
                        active_positions.append(pos)
                except (TypeError, ValueError):
                    pass
            
            if active_positions and verbose:
                logger.info("=== ACTIVE POSITIONS ===")
                
                # If verbose mode, also get open orders to show with positions
                try:
                    open_orders = self.exchange.fetch_open_orders(self.symbol)
                except Exception as e:
                    logger.warning(f"Could not fetch open orders: {e}")
                    open_orders = []
                
                for position in active_positions:
                    try:
                        position_side = position.get('side', '')
                        
                        # Use safer conversion with defaults for values that might be None
                        position_size = float(position.get('contracts', 0) or 0)
                        entry_price = float(position.get('entryPrice', 0) or 0)
                        current_price = float(position.get('markPrice', 0) or position.get('lastPrice', 0) or 0)
                        leverage = float(position.get('leverage', 1) or 1)
                        # If current_price is still 0, try to get it from ticker
                        if current_price == 0:
                            try:
                                ticker = self.exchange.fetch_ticker(self.symbol)
                                current_price = ticker['last']
                                logger.debug(f"Using ticker price {current_price} instead of position price")
                            except Exception as e:
                                logger.warning(f"Could not fetch ticker price: {e}")
                        # Calculate PnL
                        pnl_usd = float(position.get('unrealizedPnl', 0) or 0)
                        
                        # Calculate percentage PnL (accounting for leverage)
                        if entry_price > 0 and current_price > 0:  # Avoid division by zero
                            if position_side == 'long':
                                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                            else:  # short
                                price_change_pct = ((entry_price - current_price) / entry_price) * 100
                            
                            # Apply leverage to get actual PnL percentage
                            pnl_pct = price_change_pct * leverage
                            
                            # Sanity check - cap extreme values
                            pnl_pct = max(min(pnl_pct, 1000), -1000)
                        else:
                            price_change_pct = 0
                            pnl_pct = 0
                        
                        # Format direction indicator WITHOUT emojis (to avoid encoding issues)
                        direction_indicator = "LONG" if position_side == 'long' else "SHORT"
                        
                        # Log position details safely without emojis
                        logger.info(f"{direction_indicator} position: {position_size} contracts @ ${entry_price:.2f}")
                        logger.info(f"   Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | Leverage: {leverage}x")
                        logger.info(f"   PnL: {'+'if pnl_usd > 0 else ''}{pnl_usd:.2f} USD ({'+'if pnl_pct > 0 else ''}{pnl_pct:.2f}%)")
                        
                        # Add position age if available
                        if hasattr(self, 'position_opened_time') and isinstance(self.position_opened_time, datetime):
                            position_age_minutes = (datetime.now() - self.position_opened_time).total_seconds() / 60
                            hours = int(position_age_minutes // 60)
                            minutes = int(position_age_minutes % 60)
                            logger.info(f"   Age: {hours}h {minutes}m")
                        
                        # Show stop and target if available
                        if hasattr(self, 'trailing_stop_data') and self.trailing_stop_data is not None:
                            stop_price = self.trailing_stop_data.get('current_stop_price', 'N/A')
                            target_price = self.trailing_stop_data.get('take_profit_price', 'N/A')
                            stop_str = f"{stop_price:.2f}" if isinstance(stop_price, (int, float)) else str(stop_price)
                            target_str = f"{target_price:.2f}" if isinstance(target_price, (int, float)) else str(target_price)
                            logger.info(f"   Stop: ${stop_str} | Target: ${target_str}")
                    
                    except Exception as e:
                        # If we encounter any formatting errors, log with basic info
                        logger.warning(f"Error formatting position details: {e}")
                        logger.info(f"Position found: {position.get('side', 'unknown')} size: {position.get('contracts', 'unknown')}")
                    
                    # After logging position details, show related orders
                    if open_orders:
                        reduce_only_orders = [order for order in open_orders 
                                            if order.get('reduceOnly') == True]
                        
                        if reduce_only_orders:
                            logger.info("   Associated Orders:")
                            for order in reduce_only_orders:
                                try:
                                    order_type = order.get('type', '').upper()
                                    order_side = order.get('side', '').upper()
                                    order_price = float(order.get('price', 0) or 0)
                                    order_amount = float(order.get('amount', 0) or 0)
                                    
                                    # Determine if it's stop loss or take profit
                                    order_desc = "Unknown"
                                    if 'stop' in order_type.lower():
                                        order_desc = "STOP LOSS  "
                                    elif position_side == 'long' and order_side.lower() == 'sell':
                                        if order_price > entry_price:
                                            order_desc = "TAKE PROFIT"
                                        else:
                                            order_desc = "STOP LOSS  "
                                    elif position_side == 'short' and order_side.lower() == 'buy':
                                        if order_price < entry_price:
                                            order_desc = "TAKE PROFIT"
                                        else:
                                            order_desc = "STOP LOSS  "
                                    
                                    logger.info(f"   >>> {order_desc}: {order_side} {order_amount} @ ${order_price:.2f}")
                                except Exception as e:
                                    logger.warning(f"Error formatting order: {e}")
            
            return positions
            
        except Exception as e:
            log_error(e, "retrieving open positions")
            return []

    ###### UPDATED TESTING #######
    def market_exit(self):
        """
        Closes all open positions with a direct market order
        
        Returns:
        --------
        list: Executed close orders
        """
        try:
            # Get open positions first to verify if any still exist
            positions = self.get_open_positions(verbose=False)
            active_positions = []
            
            for pos in positions:
                try:
                    if float(pos.get('contracts', 0) or 0) != 0:
                        active_positions.append(pos)
                except (TypeError, ValueError):
                    pass

            if not active_positions:
                logger.info("No open positions to close")
                # Reset tracking data even if no positions exist
                self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                self.trailing_stop_data = None
                self.position_verified = False
                return []
            
            # Now we know we have active positions to close
            position = active_positions[0]
            position_size = float(position.get('contracts', 0) or 0)
            position_side = position.get('side', '')
            
            if position_size == 0:
                logger.info("Position size is zero - nothing to close")
                return []
                
            # No need to cancel the stop-loss and take-profit orders as they'll be automatically
            # cancelled by the exchange when the position is closed (since they're reduceOnly)
            
            # Just reset our tracking information
            self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                                
            # Determine close order parameters
            close_side = 'buy' if position_side == 'short' else 'sell'
            close_size = abs(position_size)
            
            logger.info(f"Closing {position_side} position of {close_size} contracts")
            
            # Get current price for reference
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            close_orders = []
            
            # Execute market order with slippage protection
            try:
                # Set a reasonable slippage buffer 
                slippage_factor = 1.005 if close_side == 'buy' else 0.995  # 0.5% slippage protection 
                slippage_price = current_price * slippage_factor
                
                # Execute the market order
                close_order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=close_side,
                    amount=close_size,
                    price=slippage_price,  # Include price for slippage calculation
                    params={"reduceOnly": True}  # Ensure it only reduces the position
                )
                
                logger.info(f"Market exit order completed: {close_order['id']}")
                close_orders.append(close_order)
                
                # Record trade with reason
                self._record_exit_trade_with_reason(position, close_order, 'market_exit')
                
            except Exception as e:
                logger.error(f"Error creating market exit order: {e}")
                
                # Try one more time with different parameters
                try:
                    # Make sure we use a valid price for the fallback attempt
                    ticker_retry = self.exchange.fetch_ticker(self.symbol)
                    retry_price = ticker_retry['last']
                    
                    slippage_factor = 1.02 if close_side == 'buy' else 0.98  # 2% slippage protection
                    slippage_price = retry_price * slippage_factor
                    
                    market_order = self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side=close_side,
                        amount=close_size,
                        price=slippage_price,  # Explicit price for slippage calculation
                        params={
                            "reduceOnly": True,
                            "slippage": 2.0  # Explicitly set 2% slippage allowance
                        }
                    )
                    
                    logger.info(f"Fallback market exit order completed: {market_order['id']}")
                    close_orders.append(market_order)
                    
                    # Record trade with reason
                    self._record_exit_trade_with_reason(position, market_order, 'market_exit')
                    
                except Exception as e2:
                    logger.error(f"All exit attempts failed: {e2}")
                    
                    # Last resort - try a limit order at a very aggressive price
                    try:
                        ticker_final = self.exchange.fetch_ticker(self.symbol)
                        final_price = ticker_final['last']
                        
                        # Use a very aggressive price to ensure execution
                        aggressive_price = final_price * 0.999 if close_side == 'sell' else final_price * 1.001
                        
                        limit_order = self.exchange.create_order(
                            symbol=self.symbol,
                            type='limit',
                            side=close_side,
                            amount=close_size,
                            price=aggressive_price,
                            params={
                                "reduceOnly": True,
                                "timeInForce": "IOC"  # Immediate-or-Cancel
                            }
                        )
                        
                        logger.info(f"Last resort limit exit order placed: {limit_order['id']}")
                        close_orders.append(limit_order)
                        
                    except Exception as e3:
                        logger.error(f"All exit mechanisms failed: {e3}")
            
            # Reset position tracking
            self.trailing_stop_data = None
            self.position_verified = False
            
            return close_orders
            
        except Exception as e:
            log_error(e, "market exit")
            return []

    ###### UPDATED TESTING #######
    def limit_chase_entry(self, side, size, max_attempts=2, timeout=4, max_total_wait=8, params=None):
        """
        Simplified market order execution that focuses on a single reliable attempt
        
        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        size : float
            Order size in contracts
        max_attempts : int
            Maximum number of attempts (kept for compatibility)
        timeout : int
            Timeout in seconds (kept for compatibility)
        max_total_wait : int
            Maximum seconds to wait (kept for compatibility)
        params : dict, optional
            Additional parameters for order execution
        
        Returns:
        --------
        dict: Executed order
        """
        if params is None:
            params = {}
        try:
            # Set a flag to indicate we're in the process of opening a position
            self.position_opening_in_progress = True

            logger.info(f"Executing {side} market order for {size} contracts")
            
            # Get current ticker for price reference
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Apply slippage protection
            # For buy orders, set slightly above market; for sell orders, slightly below
            slippage_factor = 1.003 if side == 'buy' else 0.997  # 0.3% slippage
            slippage_price = current_price * slippage_factor
            
            # Sanity check - make sure slippage price is reasonable
            if abs(slippage_price - current_price) / current_price > 0.005:
                logger.warning(f"Calculated slippage price {slippage_price} too far from current price {current_price}. Using safer value.")
                slippage_price = current_price * (1.001 if side == 'buy' else 0.999)  # Limit to 0.1% difference
                
            slippage_price = round(slippage_price, 2)  # Round to reasonable precision
            
            # Make a copy of params and ensure explicit slippage is added
            execution_params = params.copy()
            if "slippage" not in execution_params:
                execution_params["slippage"] = 0.5  # 0.5% slippage allowance
            
            # Execute market order with all required parameters for Hyperliquid
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=size,
                price=slippage_price,  # Reference price with slippage - required by Hyperliquid
                params=execution_params
            )
            
            logger.info(f"Market order executed: {order['id']} for {size} contracts at reference price {slippage_price}")
            
            # Wait for order to be processed by exchange
            time.sleep(2)
            
            # Check order status to verify fill
            try:
                order_status = self.exchange.fetch_order(order['id'], self.symbol)
                filled_amount = float(order_status.get('filled', 0) or 0)
                
                # Log fill status
                if filled_amount > 0:
                    filled_pct = (filled_amount / size) * 100
                    logger.info(f"Market order filled: {filled_amount}/{size} ({filled_pct:.1f}%)")
                else:
                    logger.warning(f"Market order may not have been filled yet: {order['id']}")
            except Exception as e:
                logger.warning(f"Error checking market order status: {e}")
            
            # Verify position was opened correctly
            try:
                verification_success = self._verify_position_update(side, [order])
                
                if not verification_success:
                    logger.warning("Initial position verification failed. Waiting 3 seconds and retrying...")
                    time.sleep(3)
                    
                    # Try verification again
                    verification_success = self._verify_position_update(side, [order])
                    
                    if not verification_success:
                        logger.error("Position verification failed after retry. Market order may not be filled yet.")
                        self.position_opening_in_progress = False  # Clear the flag
                        return None  # Return None to indicate failure
                
                # Store order execution time
                self.position_opened_time = datetime.now()
                
                return order
            except Exception as e:
                log_error(e, "verifying position update")
                self.position_opening_in_progress = False  # Clear the flag
                return None
                
        except Exception as e:
            log_error(e, "executing market order")
            self.position_opening_in_progress = False  # Clear the flag
            return None



    def _verify_position_update(self, side, filled_orders):
        """Helper method to verify position was updated correctly"""
        try:
            # Wait longer for market orders to be processed
            time.sleep(3)
            
            # Get positions after order execution
            positions = self.get_open_positions(verbose=False)
            
            # Check if position matches expected side
            expected_direction = 1 if side == 'buy' else -1
            expected_size = sum(float(order.get('filled', 0) or 0) for order in filled_orders)
            
            # Add a safety check for zero expected size
            if expected_size <= 0.001:  # Use a very small threshold instead of exactly 0
                # Try to extract size from orders directly
                for order in filled_orders:
                    if 'amount' in order and order['amount']:
                        expected_size = float(order['amount'])
                        break
                
                if expected_size <= 0.001:
                    logger.warning(f"Invalid expected position size: {expected_size}. Order may not be filled yet.")
                    return False
            
            # Also check for positions without an explicit side match - sometimes the exchange 
            # might report a position with an ambiguous side immediately after order execution
            any_position_found = False
            position_found = False
            
            for position in positions:
                try:
                    position_size = float(position.get('contracts', 0) or 0)
                    if position_size > 0.001:  # Any non-zero position
                        any_position_found = True
                    
                    position_side = position.get('side', '')
                    
                    # Check if position side and approximate size match expectations
                    if ((position_side == 'long' and expected_direction > 0) or 
                        (position_side == 'short' and expected_direction < 0)):
                        # Use a more generous tolerance for market orders (10%)
                        size_diff = abs(abs(position_size) - expected_size)
                        if size_diff <= max(0.001, expected_size * 0.1):  # 10% tolerance
                            logger.info(f"Position verification successful: {position_side} position of {position_size} contracts")
                            
                            self.position_opened_time = datetime.now()
                            self.position_verified = False
                            logger.info(f"Position opened at {self.position_opened_time}")
                            position_found = True
                            return True
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error checking position data: {e}")
                    continue
            
            if any_position_found and not position_found:
                logger.warning(f"Found position but side/size doesn't match expected: {side} / {expected_size}")
                # If any position exists and orders were filled, consider it a success despite mismatch
                if len(filled_orders) > 0 and any([order.get('status') == 'closed' for order in filled_orders]):
                    logger.info("Orders appear to be filled, accepting position despite mismatch")
                    self.position_opened_time = datetime.now()
                    self.position_verified = False
                    return True
            
            if not position_found and expected_size > 0:
                logger.warning(f"Position verification failed! Expected {expected_size} contracts {side}, but position not found or size mismatch")
            
            return False
            
        except Exception as e:
            logger.error(f"Error in position verification: {e}")
            return False
        


    ##################################################################################################################################################
    ############################################################## Stop & Trailing Stop ##############################################################
    ##################################################################################################################################################

    def set_exchange_stop_orders(self, side, entry_price, stop_loss_price, take_profit_price, position_size):
        logger.info(f"Setting exchange orders - Entry: {entry_price}, SL: {stop_loss_price}, TP: {take_profit_price}")
        """
        Set stop-loss and take-profit orders directly on the exchange
        
        Parameters:
        -----------
        side : str
            Position side ('buy' for long, 'sell' for short)
        entry_price : float
            Entry price of the position
        stop_loss_price : float
            Stop loss price
        take_profit_price : float
            Take profit price
        position_size : float
            Position size in contracts
        
        Returns:
        --------
        dict: Contains the IDs of the stop-loss and take-profit orders
        """
        # Determine the close side (opposite of entry side)
        close_side = 'sell' if side == 'buy' else 'buy'
        
        orders = {
            'stop_loss_id': None,
            'take_profit_id': None
        }
        
        try:
            # Place stop loss order
            stop_order = self.exchange.create_order(
                symbol=self.symbol,
                type='stop',  # or 'stop_limit' depending on exchange support
                side=close_side,
                amount=position_size,
                price=stop_loss_price,  # Execution price
                params={
                    "stopPrice": stop_loss_price,  # Trigger price
                    "reduceOnly": True  # Ensure it only reduces the position
                }
            )
            
            orders['stop_loss_id'] = stop_order['id']
            logger.info(f"Stop loss order placed: {stop_order['id']} at {stop_loss_price}")
            
            # Place take profit order
            tp_order = self.exchange.create_order(
                symbol=self.symbol,
                type='limit',  # or 'take_profit' depending on exchange support
                side=close_side,
                amount=position_size,
                price=take_profit_price,
                params={
                    "reduceOnly": True  # Ensure it only reduces the position
                }
            )
            
            orders['take_profit_id'] = tp_order['id']
            logger.info(f"Take profit order placed: {tp_order['id']} at {take_profit_price}")
            
            return orders
            
        except Exception as e:
            log_error(e, "setting exchange stop orders")
            return orders

    ###### UPDATED TESTING #######
    def update_exchange_stop_orders(self, orders, new_stop_loss=None, new_take_profit=None):
        """
        Update existing stop-loss and take-profit orders on the exchange
        
        Parameters:
        -----------
        orders : dict
            Contains the IDs of the stop-loss and take-profit orders
        new_stop_loss : float, optional
            New stop loss price
        new_take_profit : float, optional
            New take profit price
        
        Returns:
        --------
        dict: Updated order IDs
        """
        new_orders = orders.copy()
        positions = None
        
        try:
            # Update stop loss if provided
            if new_stop_loss is not None and orders['stop_loss_id'] is not None:
                # Get position details for the new order
                positions = self.get_open_positions()
                if not positions:
                    logger.warning("No open positions found when updating stop loss")
                    return new_orders
                
                position = positions[0]
                position_size = abs(float(position['contracts']))
                close_side = 'buy' if position['side'] == 'short' else 'sell'
                
                # First place new stop loss
                try:
                    stop_order = self.exchange.create_order(
                        symbol=self.symbol,
                        type='stop',
                        side=close_side,
                        amount=position_size,
                        price=new_stop_loss,
                        params={
                            "stopPrice": new_stop_loss,
                            "reduceOnly": True
                        }
                    )
                    
                    # Store the new order ID before canceling the old one
                    new_stop_loss_id = stop_order['id']
                    logger.info(f"Created new stop loss order: {new_stop_loss_id} at {new_stop_loss}")
                    
                    # Now cancel the old order
                    try:
                        self.exchange.cancel_order(orders['stop_loss_id'], self.symbol)
                        logger.info(f"Canceled previous stop loss order: {orders['stop_loss_id']}")
                    except Exception as e:
                        log_error(e, "canceling old stop loss order")
                        # We continue anyway since we already have the new order
                    
                    # Update the order ID in our tracking
                    new_orders['stop_loss_id'] = new_stop_loss_id
                    
                except Exception as e:
                    log_error(e, "creating new stop loss order")
            
            # Update take profit if provided
            if new_take_profit is not None and orders['take_profit_id'] is not None:
                # Get position details for the new order if not already fetched
                if not positions:
                    positions = self.get_open_positions()
                    if not positions:
                        logger.warning("No open positions found when updating take profit")
                        return new_orders
                
                position = positions[0]
                position_size = abs(float(position['contracts']))
                close_side = 'buy' if position['side'] == 'short' else 'sell'
                
                # First place new take profit
                try:
                    tp_order = self.exchange.create_order(
                        symbol=self.symbol,
                        type='limit',
                        side=close_side,
                        amount=position_size,
                        price=new_take_profit,
                        params={
                            "reduceOnly": True
                        }
                    )
                    
                    # Store the new order ID before canceling the old one
                    new_take_profit_id = tp_order['id']
                    logger.info(f"Created new take profit order: {new_take_profit_id} at {new_take_profit}")
                    
                    # Now cancel the old order
                    try:
                        self.exchange.cancel_order(orders['take_profit_id'], self.symbol)
                        logger.info(f"Canceled previous take profit order: {orders['take_profit_id']}")
                    except Exception as e:
                        log_error(e, "canceling old take profit order")
                        # We continue anyway since we already have the new order
                    
                    # Update the order ID in our tracking
                    new_orders['take_profit_id'] = new_take_profit_id
                    
                except Exception as e:
                    log_error(e, "creating new take profit order")
            
            return new_orders
            
        except Exception as e:
            log_error(e, "updating exchange stop orders")
            return new_orders

        
    def initialize_tiered_trailing_stop(self, entry_price, initial_stop_price, take_profit_price, side):
        """
        Initialize a tiered trailing stop that activates at halfway to TP
        and uses progressively tighter trailing as price approaches TP
        """
        # Calculate distance to TP and halfway point
        if side == 'long':
            tp_distance = take_profit_price - entry_price
            halfway_price = entry_price + (tp_distance * 0.55)  # 50% to TP
            initial_trail_price = entry_price + (tp_distance * 0.10)  # 15% to TP
        else:  # short
            tp_distance = entry_price - take_profit_price
            halfway_price = entry_price - (tp_distance * 0.55)  # 50% to TP
            initial_trail_price = entry_price - (tp_distance * 0.10)  # 15% to TP
        
        # Define profit tiers (percentage of distance to TP) and corresponding trail factors
        trailing_stop = {
            'active': True,
            'side': side,
            'entry_price': entry_price,
            'current_price': entry_price,
            'take_profit_price': take_profit_price,
            'halfway_price': halfway_price,
            'initial_trail_price': initial_trail_price,  # 10% to TP point
            'initial_stop_price': initial_stop_price,
            'current_stop_price': initial_stop_price,
            'tp_distance': tp_distance,
            # Profit tiers (% of distance to TP) - starts at 50%
            'profit_tiers': [0.55, 0.65, 0.75, 0.85],
            # Trail factors (% of TP distance to use as trail distance)
            'trail_factors': [0.45, 0.45, 0.4, 0.35],
            'trailing_activated': False,
            'initial_move_made': False,  # Track if we've moved to 10% level
            'current_tier': 0,
            'last_tier': -1,
            'max_favorable_excursion': 0,
            'partial_sl_placed': False,
            'last_updated': datetime.now()
        }
        
        logger.info(f"Tiered trailing stop initialized: Entry: {entry_price}, Halfway: {halfway_price}, " +
                    f"Initial trail: {initial_trail_price}")
        return trailing_stop
    
    ###### UPDATED TESTING #######
    
    def update_tiered_trailing_stop(self, trailing_stop, current_price):
        """
        Update the trailing stop based on current price, making sure stop orders are
        placed on the exchange when needed.
        
        Parameters:
        -----------
        trailing_stop : dict
            Trailing stop parameters
        current_price : float
            Current market price
        
        Returns:
        --------
        dict: Updated trailing stop parameters
        dict: Info about the update (used for logging)
        """
        # Skip update if trailing stops are disabled
        if not hasattr(self, 'use_trailing_stop') or not self.use_trailing_stop:
            return trailing_stop, {'action': 'none', 'reason': 'trailing_stop_disabled'}
            
        try:
            if trailing_stop is None:
                logger.warning("Trailing stop data is None. Creating new empty structure.")
                return None, {'action': 'none', 'reason': 'no_data'}
            
            # Check if this is an old format trailing stop and needs migration
            if 'active' not in trailing_stop or 'profit_tiers' not in trailing_stop:
                logger.warning("Trailing stop data has incorrect format. Resetting.")
                return None, {'action': 'none', 'reason': 'invalid_format'}
            
            # Skip if trailing stop is not active
            if not trailing_stop['active']:
                return trailing_stop, {'action': 'none', 'reason': 'inactive'}
            
            # Calculate price movement since entry
            side = trailing_stop['side']
            entry_price = trailing_stop['entry_price']
            take_profit_price = trailing_stop['take_profit_price']
            tp_distance = trailing_stop['tp_distance']
            favorable_move = 0
            
            if side == 'long':
                favorable_move = current_price - entry_price
            else:  # short
                favorable_move = entry_price - current_price
            
            # Ensure max_favorable_excursion exists before trying to access it
            if 'max_favorable_excursion' not in trailing_stop:
                trailing_stop['max_favorable_excursion'] = favorable_move
            else:
                # Update maximum favorable excursion if we have a new high
                prev_max = trailing_stop['max_favorable_excursion']
                trailing_stop['max_favorable_excursion'] = max(prev_max, favorable_move)
            
            # Update current price
            trailing_stop['current_price'] = current_price
            
            # Check if enough time has passed since last update (30 seconds limit)
            current_time = datetime.now()
            
            # Ensure last_updated exists
            if 'last_updated' not in trailing_stop:
                trailing_stop['last_updated'] = current_time
                update_allowed = True
            else:
                time_since_last_update = (current_time - trailing_stop['last_updated']).total_seconds()
                update_allowed = time_since_last_update >= 30  # 30 seconds 
            
            # If not yet activated, check if we should activate trailing
            if not trailing_stop.get('trailing_activated', False):
                # Ensure trailing_activated exists
                if 'trailing_activated' not in trailing_stop:
                    trailing_stop['trailing_activated'] = False
                
                # Calculate progress toward take profit (as percentage)
                if side == 'long':
                    progress_pct = (current_price - entry_price) / tp_distance
                    progress_pct_display = round(progress_pct * 100, 2)  # Convert to percentage and round
                    reached_halfway = current_price >= trailing_stop['halfway_price']
                else:  # short
                    progress_pct = (entry_price - current_price) / tp_distance
                    progress_pct_display = round(progress_pct * 100, 2)  # Convert to percentage and round
                    reached_halfway = current_price <= trailing_stop['halfway_price']                
                
                # If we've reached halfway, activate trailing
                if reached_halfway:
                    trailing_stop['trailing_activated'] = True
                    logger.info(f"Trailing stop activated at halfway point! Price: {current_price}")
                    
                    # Force update on activation regardless of time
                    update_allowed = True
                else:
                    return trailing_stop, {
                        'action': 'wait',
                        'reason': 'not_activated',
                        'progress_pct': progress_pct_display
                    }
            
            # Ensure initial_move_made exists
            if 'initial_move_made' not in trailing_stop:
                trailing_stop['initial_move_made'] = False
            
            # Step 2: First move to initial trail level if not already done
            if trailing_stop['trailing_activated'] and not trailing_stop['initial_move_made']:
                # Ensure current_stop_price exists
                if 'current_stop_price' not in trailing_stop:
                    if side == 'long':
                        trailing_stop['current_stop_price'] = entry_price  # Initialize to entry for long
                    else:
                        trailing_stop['current_stop_price'] = entry_price  # Initialize to entry for short
                
                prev_stop = trailing_stop['current_stop_price']
                trailing_stop['current_stop_price'] = trailing_stop['initial_trail_price']
                trailing_stop['initial_move_made'] = True
                
                # Update stop-loss on exchange if allowed
                if update_allowed:
                    try:
                        # Get current position
                        positions = self.get_open_positions(verbose=False)
                        if positions:
                            position = positions[0]
                            position_size = abs(float(position.get('contracts', 0)))
                            position_side = position['side']
                            
                            # Determine close side (opposite of position side)
                            close_side = 'buy' if position_side == 'short' else 'sell'
                            
                            # Create new stop loss first
                            new_stop_order = self.exchange.create_order(
                                symbol=self.symbol,
                                type='stop',
                                side=close_side,
                                amount=position_size,
                                price=trailing_stop['initial_trail_price'],
                                params={
                                    'stopPrice': trailing_stop['initial_trail_price'],
                                    'reduceOnly': True
                                }
                            )
                            
                            logger.info(f"Created new stop loss order at initial trail level: {new_stop_order['id']} @ {trailing_stop['initial_trail_price']}")
                            
                            # Cancel the old stop loss order if it exists
                            if hasattr(self, 'exchange_orders') and self.exchange_orders.get('stop_loss_id'):
                                try:
                                    self.exchange.cancel_order(self.exchange_orders['stop_loss_id'], self.symbol)
                                    logger.info(f"Canceled old stop loss: {self.exchange_orders['stop_loss_id']}")
                                except Exception as e:
                                    logger.warning(f"Error canceling old stop loss: {e}")
                            
                            # Update tracking with new order ID
                            if not hasattr(self, 'exchange_orders'):
                                self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                                
                            self.exchange_orders['stop_loss_id'] = new_stop_order['id']
                        else:
                            logger.warning("No active position found when trying to update stop loss")
                    except Exception as e:
                        log_error(e, "updating stop loss to initial trail level")
                    
                    trailing_stop['last_updated'] = current_time
                    return trailing_stop, {
                        'action': 'initial_move',
                        'prev_stop': prev_stop,
                        'new_stop': trailing_stop['initial_trail_price'],
                        'exchange_update': True
                    }
                else:
                    return trailing_stop, {
                        'action': 'initial_move_internal',
                        'prev_stop': prev_stop,
                        'new_stop': trailing_stop['initial_trail_price'],
                        'time_to_next_update': 60 - time_since_last_update,
                        'exchange_update': False
                    }
            
            # Step 3: Determine which tier the current progress falls into
            if trailing_stop.get('initial_move_made', False):
                # Calculate progress toward take profit (as percentage)
                if side == 'long':
                    progress_pct = (current_price - entry_price) / tp_distance
                else:  # short
                    progress_pct = (entry_price - current_price) / tp_distance
                
                progress_pct = min(max(0, progress_pct), 1.0)  # Clamp to 0-100%
                
                # Find appropriate tier based on current progress
                current_tier = 0
                for i, tier_threshold in enumerate(trailing_stop['profit_tiers']):
                    if progress_pct >= tier_threshold:
                        current_tier = i
                
                # Get trail factor for current tier
                trail_factor = trailing_stop['trail_factors'][current_tier]
                
                # Calculate trail distance based on TP distance and current tier's factor
                trail_distance = tp_distance * trail_factor
                
                # Calculate new stop price
                new_stop_price = 0
                if side == 'long':
                    new_stop_price = current_price - trail_distance
                else:  # short
                    new_stop_price = current_price + trail_distance
                
                # Only move stop if it improves the current stop
                if ((side == 'long' and new_stop_price > trailing_stop['current_stop_price']) or
                    (side == 'short' and new_stop_price < trailing_stop['current_stop_price'])):
                    
                    prev_stop = trailing_stop['current_stop_price']
                    trailing_stop['current_stop_price'] = new_stop_price
                    trailing_stop['current_tier'] = current_tier
                    
                    # Calculate profit locked in as percentage of TP
                    if side == 'long':
                        lock_in_pct = ((new_stop_price - entry_price) / tp_distance) * 100
                    else:
                        lock_in_pct = ((entry_price - new_stop_price) / tp_distance) * 100
                    
                    # Ensure last_tier exists
                    if 'last_tier' not in trailing_stop:
                        trailing_stop['last_tier'] = -1
                    
                    tier_change = current_tier > trailing_stop['last_tier']
                    trailing_stop['last_tier'] = current_tier
                    
                    # Update exchange stop-loss if allowed
                    if update_allowed:
                        try:
                            # Get current position
                            positions = self.get_open_positions(verbose=False)
                            if positions:
                                position = positions[0]
                                position_size = abs(float(position.get('contracts', 0)))
                                position_side = position['side']
                                
                                # Determine close side (opposite of position side)
                                close_side = 'buy' if position_side == 'short' else 'sell'
                                
                                # First create the new stop-loss order
                                new_stop_order = self.exchange.create_order(
                                    symbol=self.symbol,
                                    type='stop',
                                    side=close_side,
                                    amount=position_size,
                                    price=new_stop_price,
                                    params={
                                        'stopPrice': new_stop_price,
                                        'reduceOnly': True
                                    }
                                )
                                
                                logger.info(f"Created new stop loss order at tier {current_tier}: {new_stop_order['id']} @ {new_stop_price}")
                                
                                # Then cancel the old stop-loss order if it exists
                                if hasattr(self, 'exchange_orders') and self.exchange_orders.get('stop_loss_id'):
                                    try:
                                        self.exchange.cancel_order(self.exchange_orders['stop_loss_id'], self.symbol)
                                        logger.info(f"Canceled old stop loss order: {self.exchange_orders['stop_loss_id']}")
                                    except Exception as e:
                                        logger.warning(f"Error canceling old stop loss: {e}")
                                
                                # Update tracking with new order ID
                                if not hasattr(self, 'exchange_orders'):
                                    self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                                
                                self.exchange_orders['stop_loss_id'] = new_stop_order['id']
                            else:
                                logger.warning("No active position found when trying to update stop loss at tier change")
                        except Exception as e:
                            log_error(e, f"updating stop loss to tier {current_tier}")
                        
                        trailing_stop['last_updated'] = current_time
                        return trailing_stop, {
                            'action': 'adjusted',
                            'prev_stop': prev_stop,
                            'new_stop': new_stop_price,
                            'lock_in_pct': lock_in_pct,
                            'progress_pct': progress_pct * 100,
                            'tier': current_tier,
                            'tier_changed': tier_change,
                            'trail_factor': trail_factor,
                            'exchange_update': True
                        }
                    else:
                        return trailing_stop, {
                            'action': 'adjusted_internal',
                            'prev_stop': prev_stop,
                            'new_stop': new_stop_price,
                            'lock_in_pct': lock_in_pct,
                            'progress_pct': progress_pct * 100,
                            'tier': current_tier,
                            'tier_changed': tier_change,
                            'trail_factor': trail_factor,
                            'time_to_next_update': 60 - time_since_last_update,
                            'exchange_update': False
                        }
            
            return trailing_stop, {'action': 'none', 'reason': 'no_adjustment_needed'}
        
        except Exception as e:
            log_error(e, "updating trailing stop")
            return trailing_stop, {'action': 'error', 'reason': str(e)}    
    

    
    ###### UPDATED TESTING #######
    def check_trailing_stop_hit(self, trailing_stop, current_price):
        """
        Enhanced version that checks if the trailing stop level has been hit,
        with additional checks to prevent false triggers.
        """
        # Skip check if trailing stops are disabled
        if not hasattr(self, 'use_trailing_stop') or not self.use_trailing_stop:
            return False
        
        if trailing_stop is None or not trailing_stop.get('active', False):
            return False
        
        # Check if trailing stop is actually activated
        if not trailing_stop.get('trailing_activated', False):
            return False
        
        # NEW: Add a check for recent activation
        if 'last_updated' in trailing_stop:
            time_since_update = (datetime.now() - trailing_stop['last_updated']).total_seconds()
            # If stop level was updated less than 5 seconds ago, don't check for hits yet
            # This gives the exchange time to process the new stop order
            if time_since_update < 5:
                logger.debug(f"Trailing stop recently updated ({time_since_update:.1f}s ago) - delaying hit check")
                return False
        
        # Get the current stop level
        stop_level = trailing_stop.get('current_stop_price')
        if stop_level is None:
            logger.warning("Trailing stop is active but has no current_stop_price")
            return False
            
        side = trailing_stop.get('side')
        if side is None:
            logger.warning("Trailing stop is active but has no side information")
            return False
        
        # Convert to float and ensure valid values
        try:
            stop_level = float(stop_level)
            current_price = float(current_price)
        except (ValueError, TypeError):
            logger.warning(f"Invalid stop level or current price: stop={stop_level}, price={current_price}")
            return False
        
        # Add extra logging to help diagnose issues
        logger.debug(f"Checking trailing stop hit: side={side}, current_price={current_price}, stop_level={stop_level}")
        
        # Rather than checking manually, let's first verify if the stop order still exists
        # If the stop order doesn't exist, it might have been filled already by the exchange
        if hasattr(self, 'exchange_orders') and self.exchange_orders.get('stop_loss_id'):
            try:
                # Try to get the stop order status
                stop_order = self.exchange.fetch_order(self.exchange_orders['stop_loss_id'], self.symbol)
                if stop_order.get('status') == 'closed' or stop_order.get('status') == 'filled':
                    logger.info(f"Stop order {self.exchange_orders['stop_loss_id']} already filled by exchange")
                    return True
            except Exception as e:
                # If we get an error fetching the order, it might have been filled and removed
                if "order not found" in str(e).lower():
                    logger.info(f"Stop order {self.exchange_orders['stop_loss_id']} not found - may have been filled")
                    
                    # Double check if position is still open
                    positions = self.get_open_positions(verbose=False)
                    if not positions or not any(float(pos.get('contracts', 0) or 0) != 0 for pos in positions):
                        logger.info("Position appears to be closed - stop order likely filled")
                        return True
                else:
                    logger.warning(f"Error checking stop order status: {e}")
        
        # Manual check as backup
        # For long positions, stop hit if price <= stop level
        # For short positions, stop hit if price >= stop level
        if side == 'long' and current_price <= stop_level:
            # Add an extra confirmation check - get a fresh price
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                confirm_price = ticker['last']
                
                if confirm_price <= stop_level:
                    logger.warning(f"TRAILING STOP HIT CONFIRMED: Long position - Price {confirm_price} below stop level {stop_level}")
                    return True
                else:
                    logger.info(f"False alarm - Fresh price {confirm_price} is above stop level {stop_level}")
                    return False
            except Exception as e:
                # If we can't get a confirmation price, proceed with caution
                logger.warning(f"TRAILING STOP HIT: Long position - Price {current_price} crossed below stop level {stop_level}")
                logger.warning(f"Could not get confirmation price: {e}")
                return True
                
        elif side == 'short' and current_price >= stop_level:
            # Add an extra confirmation check - get a fresh price
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                confirm_price = ticker['last']
                
                if confirm_price >= stop_level:
                    logger.warning(f"TRAILING STOP HIT CONFIRMED: Short position - Price {confirm_price} above stop level {stop_level}")
                    return True
                else:
                    logger.info(f"False alarm - Fresh price {confirm_price} is below stop level {stop_level}")
                    return False
            except Exception as e:
                # If we can't get a confirmation price, proceed with caution
                logger.warning(f"TRAILING STOP HIT: Short position - Price {current_price} crossed above stop level {stop_level}")
                logger.warning(f"Could not get confirmation price: {e}")
                return True
        
        return False
    
    ##################################################################################################################################################
    ########################################################## Order & Position Monitoring ###########################################################
    ##################################################################################################################################################

    def start_position_monitor(self):
        """Start a separate thread for monitoring positions at higher frequency"""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.info("Position monitor already running")
            return
            
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self._position_monitor_loop)
        self.monitor_thread.daemon = True  # Make sure it terminates with main thread
        self.monitor_thread.start()
        logger.info(f"Position monitor started with {self.monitor_interval}s interval")
    
    def stop_position_monitor(self):
        """Stop the position monitoring thread"""
        self.monitor_active = False
        if self.monitor_thread is not None:
            logger.info("Stopping position monitor")
            # Thread will terminate in next iteration due to monitor_active flag
    

        def _position_monitor_loop(self):
            """Modified monitoring loop using WebSocket data"""
            while self.monitor_active:
                try:
                    # Skip if position is being opened
                    if hasattr(self, 'position_opening_in_progress') and self.position_opening_in_progress:
                        time.sleep(self.monitor_interval)
                        continue
                    
                    # Get position data from WebSocket instead of API
                    ws_positions = self.ws_client.get_position_data()
                    
                    if ws_positions is not None:
                        # Process WebSocket position data
                        active_position_exists = self._process_ws_positions(ws_positions)
                    else:
                        # Fallback to API if WebSocket data is not available
                        positions = self._throttled_api_call(
                            "get_positions", 
                            self.exchange.fetch_positions,
                            [self.symbol]
                        )
                        
                        if positions is not None:
                            active_position_exists = positions and any(float(pos.get('contracts', 0) or 0) != 0 for pos in positions)
                        else:
                            # If API also fails, wait and retry
                            time.sleep(self.monitor_interval * 2)
                            continue
                    
                    if not active_position_exists:
                        logger.info("No active positions, exiting monitor")
                        break
                    
                    # Get order data from WebSocket
                    ws_orders = self.ws_client.get_order_data()
                    
                    if ws_orders is not None:
                        # Process WebSocket order data
                        self._process_ws_orders(ws_orders)
                    else:
                        # Only fetch via API occasionally as backup
                        if (time.time() - self.last_order_api_check) > 120:  # Every 2 minutes
                            orders = self._throttled_api_call(
                                "fetch_open_orders",
                                self.exchange.fetch_open_orders,
                                self.symbol
                            )
                            if orders is not None:
                                self._verify_orders_from_data(self.position_cache[0], orders, self.price_cache)
                                self.last_order_api_check = time.time()
                    
                    # Check for stop hits with cached data
                    if self.price_cache and self.trailing_stop_data:
                        if self.check_trailing_stop_hit(self.trailing_stop_data, self.price_cache):
                            logger.warning(f"Stop hit at {self.price_cache}! Executing market exit")
                            self.market_exit()
                            break
                    
                    time.sleep(self.monitor_interval)
                    
                except Exception as e:
                    log_error(e, "position monitor loop")
                    time.sleep(self.monitor_interval * 2)

    # Add this new helper method
    def _verify_orders_from_data(self, position, open_orders, current_price):
        """Verify and track orders without making additional API calls"""
        try:
            position_side = position.get('side', '')
            position_size = float(position.get('contracts', 0) or 0)
            entry_price = float(position.get('entryPrice', 0) or 0)
            
            # Determine close side (opposite of position)
            close_side = 'sell' if position_side == 'long' else 'buy'
            
            # Initialize tracking if needed
            if not hasattr(self, 'exchange_orders'):
                self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
            
            # Track found orders
            found_stop_loss = False
            found_take_profit = False
            
            # Check our tracked orders first
            for order in open_orders:
                if order.get('id') == self.exchange_orders.get('stop_loss_id'):
                    found_stop_loss = True
                elif order.get('id') == self.exchange_orders.get('take_profit_id'):
                    found_take_profit = True
            
            # If we didn't find our tracked orders, look for matching ones
            if not found_stop_loss or not found_take_profit:
                for order in open_orders:
                    order_side = order.get('side', '')
                    if order_side != close_side:
                        continue
                    
                    is_reduce_only = order.get('reduceOnly', False)
                    if not is_reduce_only:
                        continue
                    
                    order_type = order.get('type', '').lower()
                    order_price = float(order.get('price', 0) or 0)
                    
                    # Identify stop loss orders
                    if not found_stop_loss and ('stop' in order_type or order.get('stopPrice')):
                        found_stop_loss = True
                        self.exchange_orders['stop_loss_id'] = order.get('id')
                        logger.info(f"Found stop loss order: {order.get('id')}")
                    
                    # Identify take profit orders (limit orders above entry for longs, below for shorts)
                    elif not found_take_profit and order_type == 'limit':
                        if (position_side == 'long' and order_price > entry_price) or \
                        (position_side == 'short' and order_price < entry_price):
                            found_take_profit = True
                            self.exchange_orders['take_profit_id'] = order.get('id')
                            logger.info(f"Found take profit order: {order.get('id')}")
            
            # Create missing orders if needed, but be careful with API calls
            # Only do this during full verification checks
            if (not found_stop_loss or not found_take_profit) and not self.rate_limited:
                # Calculate stop loss and take profit levels
                if hasattr(self, 'trailing_stop_data') and self.trailing_stop_data is not None:
                    stop_loss = self.trailing_stop_data.get('current_stop_price')
                    take_profit = self.trailing_stop_data.get('take_profit_price')
                else:
                    # Calculate based on ATR or fixed percentage
                    if hasattr(self, 'data') and self.data is not None and 'atr' in self.data.columns:
                        atr = self.data['atr'].iloc[-1]
                    else:
                        atr = entry_price * 0.0075  # Fallback
                    
                    if position_side == 'long':
                        stop_loss = entry_price - (atr * self.stop_loss_atr_multiple)
                        take_profit = entry_price + (atr * self.take_profit_atr_multiple)
                    else:
                        stop_loss = entry_price + (atr * self.stop_loss_atr_multiple)
                        take_profit = entry_price - (atr * self.take_profit_atr_multiple)
                
                # Create missing stop loss order
                if not found_stop_loss:
                    logger.warning(f"Stop loss order not found - creating new one at {stop_loss}")
                    try:
                        stop_order = self._throttled_api_call(
                            "create_order_sl",
                            self.exchange.create_order,
                            symbol=self.symbol,
                            type='stop',
                            side=close_side,
                            amount=position_size,
                            price=stop_loss,
                            params={
                                "stopPrice": stop_loss,
                                "reduceOnly": True
                            }
                        )
                        
                        if stop_order:
                            self.exchange_orders['stop_loss_id'] = stop_order['id']
                            logger.info(f"Created new stop loss order: {stop_order['id']} at {stop_loss}")
                    except Exception as e:
                        logger.error(f"Failed to create stop loss: {e}")
                
                # Create missing take profit order
                if not found_take_profit:
                    logger.warning(f"Take profit order not found - creating new one at {take_profit}")
                    try:
                        tp_order = self._throttled_api_call(
                            "create_order_tp",
                            self.exchange.create_order,
                            symbol=self.symbol,
                            type='limit',
                            side=close_side,
                            amount=position_size,
                            price=take_profit,
                            params={"reduceOnly": True}
                        )
                        
                        if tp_order:
                            self.exchange_orders['take_profit_id'] = tp_order['id']
                            logger.info(f"Created new take profit order: {tp_order['id']} at {take_profit}")
                    except Exception as e:
                        logger.error(f"Failed to create take profit: {e}")
            
            # Mark position as verified
            self.position_verified = True
            
            return found_stop_loss and found_take_profit
        
        except Exception as e:
            logger.error(f"Error verifying orders from data: {e}")
            return False


    ###### UPDATED TESTING #######

    def calculate_emergency_levels(self, position):
        """
        Calculate emergency stop loss and take profit levels using fixed percentages
        
        Parameters:
        -----------
        position : dict
            Position details from the exchange
        
        Returns:
        --------
        tuple: (stop_loss_price, take_profit_price)
        """
        try:
            # Extract position details
            position_side = position.get('side', '')
            entry_price = float(position.get('entryPrice', 0) or 0)
            
            if entry_price == 0:
                logger.error("Invalid entry price in position data")
                return None, None
            
            # Fixed percentages
            emergency_sl_percent = 0.75  # 0.75% stop loss
            emergency_tp_percent = 0.5   # 0.5% take profit
            
            if position_side == 'long':
                # For long positions: stop loss below entry, take profit above
                stop_loss = entry_price * (1 - emergency_sl_percent / 100)
                take_profit = entry_price * (1 + emergency_tp_percent / 100)
            elif position_side == 'short':
                # For short positions: stop loss above entry, take profit below
                stop_loss = entry_price * (1 + emergency_sl_percent / 100)
                take_profit = entry_price * (1 - emergency_tp_percent / 100)
            else:
                logger.error(f"Invalid position side: {position_side}")
                return None, None
            
            logger.info(f"Calculated emergency levels: SL={stop_loss:.4f}, TP={take_profit:.4f} " + 
                    f"(SL={emergency_sl_percent}%, TP={emergency_tp_percent}%)")
            
            return stop_loss, take_profit
        
        except Exception as e:
            logger.error(f"Error calculating emergency levels: {e}")
            return None, None

    def monitor_exit_orders(self):
        """
        Continuously monitor the status of exit orders with improved delay handling
        and proper handling of trailing stop states.
        """
        try:
            # CRITICAL: Skip all monitoring if we're in the process of opening a position
            if hasattr(self, 'position_opening_in_progress') and self.position_opening_in_progress:
                logger.debug("Position opening in progress - skipping monitoring")
                return False
            
            # Initialize time_since_opened with a default value
            time_since_opened = 0

            # ENHANCED: Add more robust checking of position age before monitoring
            if hasattr(self, 'position_opened_time') and self.position_opened_time:
                time_since_opened = (datetime.now() - self.position_opened_time).total_seconds()
                
                # If position was opened less than 30 seconds ago, completely skip monitoring
                if time_since_opened < 10:  # Increased from original delay
                    logger.info(f"Position opened only {time_since_opened:.1f}s ago - too early for order monitoring")
                    return False
                
                # For positions between 30-120 seconds old, only check for emergency conditions
                # but don't verify or create orders yet
                elif time_since_opened < 5:  #  "soft delay" for new positions 
                    # Just check for price being far beyond our intended stops
                    positions = self.get_open_positions(verbose=False)
                    if not positions or len(positions) == 0:
                        return False
                        
                    position = positions[0]
                    position_side = position.get('side', '')
                    
                    # Get current price to check for extreme moves
                    try:
                        ticker = self.exchange.fetch_ticker(self.symbol)
                        current_price = ticker['last']
                        
                        # Use original entry and stop loss for emergency check
                        entry_price = float(position.get('entryPrice', 0) or 0)
                        
                        # For long positions, emergency stop would be far below entry
                        if position_side == 'long':
                            # Use a very wide emergency stop
                            emergency_stop = entry_price * 0.99  # 2% below entry
                            if current_price <= emergency_stop:
                                logger.warning(f"EMERGENCY EXIT: Price far below entry during delay period")
                                return True  # Trigger intervention
                        else:  # short
                            # For shorts, emergency stop would be far above entry
                            emergency_stop = entry_price * 1.01  # 2% above entry
                            if current_price >= emergency_stop:
                                logger.warning(f"EMERGENCY EXIT: Price far above entry during delay period")
                                return True  # Trigger intervention
                    except Exception as e:
                        logger.warning(f"Error checking emergency conditions during delay period: {e}")
                    
                    # For normal conditions during delay period, just skip verification
                    logger.info(f"Position opened {time_since_opened:.1f}s ago - in delay period, skipping order verification")
                    return False
            
            # Skip if no position
            positions = self.get_open_positions(verbose=False)
            
            active_positions = []
            for pos in positions:
                try:
                    if float(pos.get('contracts', 0) or 0) != 0:
                        active_positions.append(pos)
                except (TypeError, ValueError):
                    pass
            
            if not active_positions:
                # No active positions, reset order tracking
                self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                self.trailing_stop_data = None
                self.position_verified = False
                return False
                    
            # Get current position details
            position = active_positions[0]
            position_side = position.get('side', '')
            
            try:
                position_size = float(position.get('contracts', 0) or 0)
                entry_price = float(position.get('entryPrice', 0) or 0)
            except (TypeError, ValueError):
                position_size = 0
                entry_price = 0
                logger.warning("Could not parse position size or entry price")
            
            # Get current market price
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
            except Exception as e:
                log_error(e, "fetching current price")
                return False
            
            # *** NEW SECTION: Check if trailing stop should be activated ***
            # If we have trailing stop data but it's not yet activated, check if we've reached activation point
            if self.trailing_stop_data is not None and not self.trailing_stop_data.get('trailing_activated', False):
                try:
                    side = self.trailing_stop_data.get('side')
                    entry_price = self.trailing_stop_data.get('entry_price')
                    halfway_price = self.trailing_stop_data.get('halfway_price')
                    
                    # Check if we've reached the halfway point (activation trigger)
                    if side == 'long' and current_price >= halfway_price:
                        logger.info(f"Long position reached halfway point ({halfway_price}). Activating trailing stop.")
                        # Update trailing stop to activate it
                        self.trailing_stop_data, update_info = self.update_tiered_trailing_stop(self.trailing_stop_data, current_price)
                        
                        # Log the activation
                        if update_info.get('action') in ['adjusted', 'initial_move'] and update_info.get('exchange_update', False):
                            logger.info(f"Trailing stop activated! Stop moved to: {update_info.get('new_stop')}")
                        
                    elif side == 'short' and current_price <= halfway_price:
                        logger.info(f"Short position reached halfway point ({halfway_price}). Activating trailing stop.")
                        # Update trailing stop to activate it
                        self.trailing_stop_data, update_info = self.update_tiered_trailing_stop(self.trailing_stop_data, current_price)
                        
                        # Log the activation
                        if update_info.get('action') in ['adjusted', 'initial_move'] and update_info.get('exchange_update', False):
                            logger.info(f"Trailing stop activated! Stop moved to: {update_info.get('new_stop')}")
                except Exception as e:
                    logger.warning(f"Error checking trailing stop activation: {e}")
            
            # *** NEW SECTION: Update trailing stop if already activated ***
            # If trailing stop is already activated, update it periodically
            elif self.trailing_stop_data is not None and self.trailing_stop_data.get('trailing_activated', True):
                try:
                    # Only update every 60 seconds (or whatever reasonable interval)
                    time_since_last_update = float('inf')  # Default to infinity
                    if 'last_updated' in self.trailing_stop_data:
                        time_since_last_update = (datetime.now() - self.trailing_stop_data['last_updated']).total_seconds()
                        
                    # If it's been more than 60 seconds since last update, update the trailing stop
                    if time_since_last_update >= 60:
                        self.trailing_stop_data, update_info = self.update_tiered_trailing_stop(self.trailing_stop_data, current_price)
                        
                        # Log if stop was moved
                        if update_info.get('action') in ['adjusted', 'initial_move'] and update_info.get('exchange_update', False):
                            new_stop = update_info.get('new_stop')
                            tier = update_info.get('tier', 0)
                            logger.info(f"Trailing stop updated to: {new_stop} (tier: {tier})")
                except Exception as e:
                    logger.warning(f"Error updating trailing stop: {e}")
            
            # Get stop and take profit levels from our memory
            stop_loss = None
            take_profit = None
            intervention_needed = False
            intervention_reason = None
            
            # Step 1: Get the stop loss and take profit from exchange orders first
            if hasattr(self, 'exchange_orders'):
                # Check for stop loss
                if self.exchange_orders.get('stop_loss_id'):
                    try:
                        stop_order = self.exchange.fetch_order(self.exchange_orders['stop_loss_id'], self.symbol)
                        stop_price = stop_order.get('stopPrice')
                        price = stop_order.get('price')
                        stop_loss = float(stop_price or price or 0)
                    except Exception as e:
                        pass
                
                # Check for take profit - direct access by order ID
                if self.exchange_orders.get('take_profit_id'):
                    try:
                        tp_order = self.exchange.fetch_order(self.exchange_orders['take_profit_id'], self.symbol)
                        take_profit = float(tp_order.get('price', 0) or 0)
                    except Exception as e:
                        pass
            
            # Step 2: Check trailing stop data if needed
            if (stop_loss is None or take_profit is None) and self.trailing_stop_data is not None:
                try:
                    # Get appropriate stop loss based on activation state
                    if stop_loss is None:
                        if self.trailing_stop_data.get('trailing_activated', False):
                            stop_loss = self.trailing_stop_data.get('current_stop_price')
                        else:
                            stop_loss = self.trailing_stop_data.get('initial_stop_price')
                    
                    # Get take profit 
                    if take_profit is None:
                        take_profit = self.trailing_stop_data.get('take_profit_price')
                except Exception as e:
                    pass
            
            # Step 3: Search open orders as last resort
            if stop_loss is None or take_profit is None:
                try:
                    open_orders = self.exchange.fetch_open_orders(self.symbol)
                    
                    for order in open_orders:
                        order_type = order.get('type', '').lower()
                        order_side = order.get('side', '').lower()
                        order_price = float(order.get('price', 0) or 0)
                        is_reduce_only = order.get('reduceOnly', False)
                        
                        # Find take profit orders
                        if order_type == 'limit' and is_reduce_only and take_profit is None:
                            opposite_side = 'sell' if position_side == 'long' else 'buy'
                            if order_side == opposite_side:
                                # For longs, take profit is above entry
                                # For shorts, take profit is below entry
                                is_valid_tp = False
                                if position_side == 'long' and order_price > entry_price:
                                    is_valid_tp = True 
                                elif position_side == 'short' and order_price < entry_price:
                                    is_valid_tp = True
                                
                                if is_valid_tp:
                                    take_profit = order_price
                                    # Update tracking
                                    self.exchange_orders['take_profit_id'] = order.get('id')
                        
                        # Find stop loss orders
                        elif ('stop' in order_type or order.get('stopPrice')) and is_reduce_only and stop_loss is None:
                            opposite_side = 'sell' if position_side == 'long' else 'buy'
                            if order_side == opposite_side:
                                stop_price = order.get('stopPrice')
                                price = order.get('price')
                                potential_stop = float(stop_price or price or 0)
                                
                                # Verify it's valid for our position
                                is_valid_stop = False
                                if position_side == 'long' and potential_stop < entry_price:
                                    is_valid_stop = True
                                elif position_side == 'short' and potential_stop > entry_price:
                                    is_valid_stop = True
                                    
                                if is_valid_stop:
                                    stop_loss = potential_stop
                                    # Update tracking
                                    self.exchange_orders['stop_loss_id'] = order.get('id')
                except Exception as e:
                    pass
            
            # Step 4: If we STILL don't have stop and take profit, calculate using ATR or percentage
            try:
                # If we STILL don't have stop and take profit, calculate using fixed percentage
                if stop_loss is None or take_profit is None:
                    logger.warning("Could not find stop loss or take profit, calculating emergency levels")
                    
                    # Use the active position to calculate emergency levels
                    if active_positions and len(active_positions) > 0:
                        position = active_positions[0]
                        emergency_sl, emergency_tp = self.calculate_emergency_levels(position)
                        
                        if emergency_sl is not None:
                            if stop_loss is None:
                                stop_loss = emergency_sl
                                logger.info(f"Using emergency stop loss: {stop_loss}")
                        
                        if emergency_tp is not None:
                            if take_profit is None:
                                take_profit = emergency_tp
                                logger.info(f"Using emergency take profit: {take_profit}")
                    
                    # If still no values, use last resort fallback
                    if stop_loss is None or take_profit is None:
                        logger.error("Could not calculate emergency levels, using last resort fallback")
                        entry_price = float(position.get('entryPrice', current_price))
                        
                        if position_side == 'long':
                            if stop_loss is None:
                                stop_loss = entry_price * 0.99  # 1% below
                            if take_profit is None:
                                take_profit = entry_price * 1.005  # 1% above
                        else:  # short
                            if stop_loss is None:
                                stop_loss = entry_price * 1.01  # 1% above
                            if take_profit is None:
                                take_profit = entry_price * 0.995  # 1% below
                        
                        logger.info(f"Last resort levels: SL={stop_loss}, TP={take_profit}")
            except Exception as e:
                logger.error(f"Error calculating emergency stop levels: {e}")
                return False
                    
            # Check if price has moved beyond our stop or take profit levels (intervention needed)
            if position_side == 'long':
                # For long positions
                if current_price <= stop_loss:
                    # This is a signal that stop loss should be triggered - not an invalid stop
                    intervention_needed = True
                    intervention_reason = "price_below_stop"
                    logger.warning(f"Long position: Price ({current_price}) is below stop-loss ({stop_loss}). Intervention needed.")
                elif current_price >= take_profit:
                    intervention_needed = True
                    intervention_reason = "price_above_take_profit"
                    logger.info(f"Price ({current_price}) is above take-profit ({take_profit}) for long position. Intervention needed.")
                    
            else:  # short
                # For short positions
                if current_price >= stop_loss:
                    # This is a signal that stop loss should be triggered - not an invalid stop
                    intervention_needed = True
                    intervention_reason = "price_above_stop"
                    logger.warning(f"Short position: Price ({current_price}) is above stop-loss ({stop_loss}). Intervention needed.")
                elif current_price <= take_profit:
                    intervention_needed = True
                    intervention_reason = "price_below_take_profit"
                    logger.info(f"Price ({current_price}) is below take-profit ({take_profit}) for short position. Intervention needed.")
            
            # If intervention is needed, execute manual exit
            if intervention_needed:
                # First check if position still exists (might have been auto-closed by exchange)
                positions_check = self.get_open_positions(verbose=False)
                if not positions_check or not any(float(pos.get('contracts', 0) or 0) != 0 for pos in positions_check):
                    logger.info("Position already closed - no intervention needed")
                    self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                    self.trailing_stop_data = None
                    self.position_verified = False
                    return False
                
                # Log the reason for intervention
                if "stop" in intervention_reason:
                    logger.info(f"Executing emergency stop-loss exit. Current price: {current_price}, Stop level: {stop_loss}")
                else:
                    logger.info(f"Executing manual take-profit exit. Current price: {current_price}, Take profit level: {take_profit}")
                
                # Use limit chase for smoother execution
                close_side = 'buy' if position_side == 'short' else 'sell'
                
                # Try limit chase for better execution
                try:
                    # Note: we're reusing limit_chase_entry but for exit purposes
                    close_order = self.limit_chase_entry(close_side, position_size, max_attempts=2, timeout=2, params={"reduceOnly": True})
                    
                    if close_order:
                        logger.info(f"Successfully closed position with limit chase: {close_order['id']}")
                        
                        # Record trade in history with correct reason
                        exit_reason = 'stop_loss' if 'stop' in intervention_reason else 'take_profit'
                        self._record_exit_trade_with_reason(position, close_order, exit_reason)
                        
                        # Reset all order tracking - no need to cancel orders as they'll be auto-cancelled by the exchange
                        self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                        self.trailing_stop_data = None
                        self.position_verified = False
                        return True
                except Exception as e:
                    log_error(e, "limit chase exit")
                    # Fallback to market exit if limit chase fails
                
                # If limit chase failed or didn't complete, use market exit as fallback
                try:
                    # Check one more time if position still exists
                    positions_check = self.get_open_positions(verbose=False)
                    if not positions_check or not any(float(pos.get('contracts', 0) or 0) != 0 for pos in positions_check):
                        logger.info("Position was closed between intervention checks - no further action needed")
                        self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                        self.trailing_stop_data = None
                        self.position_verified = False
                        return True
                    
                    close_orders = self.market_exit()
                    logger.info(f"Executed market exit as fallback, orders: {len(close_orders)}")
                    
                    # Reset all order tracking - no need to cancel orders as they'll be auto-cancelled by the exchange
                    self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                    self.trailing_stop_data = None
                    self.position_verified = False
                    return True
                except Exception as e:
                    log_error(e, "market exit fallback")
                    return False
            
            return False
        
        except Exception as e:
            log_error(e, "monitoring exit orders")
            return False
    
    def verify_and_track_all_orders(self):
        """
        Comprehensive verification and tracking of all open orders related to current position.
        This method ensures that stop-loss, take-profit and any partial exit orders are
        properly tracked and maintained.
        
        Returns:
        --------
        dict: Status of order verification and tracking
        """
        try:
            # Get open positions
            positions = self.get_open_positions(verbose=False)
            
            # If no positions, reset order tracking
            active_positions = []
            for pos in positions:
                try:
                    if float(pos.get('contracts', 0) or 0) != 0:
                        active_positions.append(pos)
                except (TypeError, ValueError):
                    pass
            
            if not active_positions:
                if hasattr(self, 'exchange_orders') and (self.exchange_orders.get('stop_loss_id') or self.exchange_orders.get('take_profit_id')):
                    logger.info("No active positions but order tracking exists. Resetting order tracking.")
                    self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                
                self.trailing_stop_data = None
                self.position_verified = False
                
                return {
                    'status': 'no_position',
                    'action_taken': 'reset_tracking'
                }
            
            # Get current position details
            position = active_positions[0]
            position_side = position.get('side', '')
            
            try:
                position_size = float(position.get('contracts', 0) or 0)
                entry_price = float(position.get('entryPrice', 0) or 0)
            except (TypeError, ValueError):
                position_size = 0
                entry_price = 0
                logger.warning("Could not parse position size or entry price")
            
            # Get all open orders
            try:
                open_orders = self.exchange.fetch_open_orders(self.symbol)
                logger.info(f"Verifying orders - Found {len(open_orders)} open orders for position {position_side}")
            except Exception as e:
                logger.error(f"Error fetching open orders: {e}")
                open_orders = []
            
            # Prepare tracking dictionaries
            found_orders = {
                'stop_loss': None,
                'take_profit': None,
                'partial_exits': []
            }
            
            # Initialize if not exists
            if not hasattr(self, 'exchange_orders'):
                self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
            
            # Determine close side (opposite of position)
            close_side = 'sell' if position_side == 'long' else 'buy'
            
            # First pass: check for our tracked orders by ID
            for order in open_orders:
                try:
                    order_id = order.get('id')
                    
                    # Is this our tracked stop-loss?
                    if self.exchange_orders.get('stop_loss_id') and order_id == self.exchange_orders['stop_loss_id']:
                        found_orders['stop_loss'] = order
                        logger.info(f"Found tracked stop-loss order: {order_id}")
                        
                    # Is this our tracked take-profit?
                    if self.exchange_orders.get('take_profit_id') and order_id == self.exchange_orders['take_profit_id']:
                        found_orders['take_profit'] = order
                        logger.info(f"Found tracked take-profit order: {order_id}")
                except Exception as e:
                    logger.warning(f"Error processing order in first pass: {e}")
                    continue
            
            # Second pass: Identify stop-loss and take-profit by characteristics if not found by ID
            if not found_orders['stop_loss'] or not found_orders['take_profit']:
                for order in open_orders:
                    try:
                        # Skip if already identified
                        if order.get('id') in [
                            self.exchange_orders.get('stop_loss_id'),
                            self.exchange_orders.get('take_profit_id')
                        ]:
                            continue
                        
                        # Look for essential order properties
                        is_reduce_only = order.get('reduceOnly', False)
                        order_side = order.get('side', '')
                        order_type = order.get('type', '')
                        
                        # Safely convert price-related values
                        order_price = float(order.get('price', 0) or 0)
                        order_stop_price = float(order.get('stopPrice', 0) or 0) if 'stopPrice' in order else 0
                        order_amount = float(order.get('amount', 0) or 0)
                        
                        # Is this a potential stop-loss order?
                        if (not found_orders['stop_loss'] and
                            is_reduce_only and 
                            order_side == close_side):
                            
                            # Check for stop order types
                            is_stop_order = False
                            if 'stop' in order_type.lower() or order_type in ['stop', 'stop_limit']:
                                is_stop_order = True
                            # Some exchanges use different indicators for stop orders
                            elif 'stopPrice' in order or order.get('stopPrice', 0) > 0:
                                is_stop_order = True
                            
                            if is_stop_order:
                                found_orders['stop_loss'] = order
                                self.exchange_orders['stop_loss_id'] = order.get('id')
                                logger.info(f"Identified stop-loss order by characteristics: {order.get('id')}")
                            
                        # Is this a potential take-profit order?
                        elif (not found_orders['take_profit'] and 
                            is_reduce_only and 
                            order_side == close_side and 
                            (order_type == 'limit' or 'take_profit' in order_type.lower())):
                            
                            # For long positions, take-profit is above entry
                            # For short positions, take-profit is below entry
                            is_take_profit = False
                            
                            if position_side == 'long' and order_price > entry_price:
                                is_take_profit = True
                            elif position_side == 'short' and order_price < entry_price:
                                is_take_profit = True
                            
                            if is_take_profit:
                                found_orders['take_profit'] = order
                                self.exchange_orders['take_profit_id'] = order.get('id')
                                logger.info(f"Identified take-profit order by characteristics: {order.get('id')}")
                                
                        # Is this a partial exit order?
                        elif (is_reduce_only and 
                            order_side == close_side and 
                            order_amount < position_size):
                            
                            found_orders['partial_exits'].append(order)
                            logger.info(f"Found partial exit order: {order.get('id')} for {order_amount} out of {position_size}")
                    
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Error processing order {order.get('id')}: {e}")
                        continue
            
            # Determine if we need to create missing orders
            need_stop_loss = not found_orders['stop_loss']
            need_take_profit = not found_orders['take_profit']
            
            # If missing orders, get current price and ATR to calculate levels
            if need_stop_loss or need_take_profit:
                # Get current price
                try:
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    current_price = ticker['last']
                except Exception as e:
                    logger.error(f"Error fetching current price: {e}")
                    current_price = entry_price  # Fall back to entry price
                
                # Get ATR for stop placement
                try:
                    if hasattr(self, 'data') and self.data is not None and len(self.data) > 0 and 'atr' in self.data.columns:
                        atr_value = self.data['atr'].iloc[-1]
                    else:
                        # Fall back to percentage of price
                        atr_value = entry_price * 0.02  # 2% as fallback
                except Exception as e:
                    logger.warning(f"Error getting ATR, using fallback: {e}")
                    atr_value = entry_price * 0.02
                
                # Calculate stop loss and take profit
                if position_side == 'long':
                    stop_loss = entry_price - (atr_value * self.stop_loss_atr_multiple)
                    take_profit = entry_price + (atr_value * self.take_profit_atr_multiple)
                    logger.info(f"Long position - Entry: {entry_price}, SL: {stop_loss} (distance: {entry_price - stop_loss}), TP: {take_profit} (distance: {take_profit - entry_price})")
                else:  # short
                    stop_loss = entry_price + (atr_value * self.stop_loss_atr_multiple)
                    take_profit = entry_price - (atr_value * self.take_profit_atr_multiple)
                    logger.info(f"Short position - Entry: {entry_price}, SL: {stop_loss} (distance: {stop_loss - entry_price}), TP: {take_profit} (distance: {entry_price - take_profit})")
                
                # Check if we have trailing stop data
                if self.trailing_stop_data is not None and self.trailing_stop_data.get('active', False):
                    try:
                        # Use trailing stop level instead of calculated one
                        stop_loss = self.trailing_stop_data.get('current_stop_price')
                        take_profit = self.trailing_stop_data.get('take_profit_price')
                    except Exception as e:
                        logger.warning(f"Error accessing trailing stop data: {e}")
                
                # Create stop-loss if needed
                if need_stop_loss:
                    logger.warning(f"Stop-loss order not found! Creating new stop-loss at {stop_loss}")
                    try:
                        logger.info("Attempting to create stop-loss order now...")
                        stop_order = self.exchange.create_order(
                            symbol=self.symbol,
                            type='stop',
                            side=close_side,
                            amount=position_size,
                            price=stop_loss,
                            params={
                                "stopPrice": stop_loss,
                                "reduceOnly": True
                            }
                        )
                        self.exchange_orders['stop_loss_id'] = stop_order['id']
                        logger.info(f"Created new stop-loss order: {stop_order['id']} at {stop_loss}")
                    except Exception as e:
                        logger.error(f"FAILED to create stop-loss order: {e}")
                        log_error(e, "creating stop-loss order")
                
                # Create take-profit if needed
                if need_take_profit:
                    logger.warning(f"Take-profit order not found! Creating new take-profit at {take_profit}")
                    try:
                        logger.info("Attempting to create take-profit order now...")
                        tp_order = self.exchange.create_order(
                            symbol=self.symbol,
                            type='limit',
                            side=close_side,
                            amount=position_size,
                            price=take_profit,
                            params={
                                "reduceOnly": True
                            }
                        )
                        self.exchange_orders['take_profit_id'] = tp_order['id']
                        logger.info(f"Created new take-profit order: {tp_order['id']} at {take_profit}")
                    except Exception as e:
                        logger.error(f"FAILED to create take-profit order: {e}")
                        log_error(e, "creating take-profit order")
            
            # Mark position as verified
            self.position_verified = True
            
            # Return verification result
            return {
                'status': 'verified',
                'stop_loss_found': found_orders['stop_loss'] is not None,
                'take_profit_found': found_orders['take_profit'] is not None,
                'partial_exits_found': len(found_orders['partial_exits']),
                'stop_loss_created': need_stop_loss,
                'take_profit_created': need_take_profit
            }
            
        except Exception as e:
            log_error(e, "verifying and tracking orders")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _record_exit_trade_with_reason(self, position, exit_order, exit_reason):
        """
        Enhanced version of _record_exit_trade that includes the specific exit reason
        
        Parameters:
        -----------
        position : dict
            Position details
        exit_order : dict
            Order details
        exit_reason : str
            Reason for exit ('stop_loss', 'take_profit', etc.)
        """
        try:
            entry_price = float(position['entryPrice'])
            position_side = position['side']
            
            # Try to get the exit price from the order, fall back to market price if needed
            if 'price' in exit_order and exit_order['price']:
                exit_price = float(exit_order['price'])
            elif 'average' in exit_order and exit_order['average']:
                exit_price = float(exit_order['average'])
            else:
                # Fall back to current market price
                ticker = self.exchange.fetch_ticker(self.symbol)
                exit_price = ticker['last']
            
            # Calculate PnL
            if position_side == 'long':
                pnl = (exit_price / entry_price - 1)
            else:
                pnl = (entry_price / exit_price - 1)
                
            self.trade_history.append({
                'entry_time': self.position_opened_time,
                'exit_time': datetime.now(),
                'position': position_side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': exit_reason
            })
            
            logger.info(f"Trade recorded - {position_side}: {entry_price} -> {exit_price}, PnL: {pnl:.2%}, Reason: {exit_reason}")
            
        except Exception as e:
            log_error(e, "recording exit trade with reason")

    ##################################################################################################################################################
    ########################################################## Trade Management & Logic ##############################################################
    ##################################################################################################################################################

    ###### UPDATED TESTING #######
    def check_exit_conditions(self):
        """
        Enhanced check for exit conditions that works alongside the new monitoring system.
        This checks additional exit conditions beyond just price reaching target levels.
        """
        positions = self.get_open_positions(verbose=False)
        
        # If no positions, reset trailing stop data
        if not positions or not any(float(pos.get('contracts', 0)) != 0 for pos in positions):
            self.trailing_stop_data = None
            return
        
        # Get position details
        position = positions[0]
        position_size = float(position.get('contracts', 0))
        position_side = position.get('side', '')
        entry_price = float(position.get('entryPrice', 0))
        
        # Get current price from data or fetch ticker
        current_price = None
        if hasattr(self, 'data') and self.data is not None and len(self.data) > 0:
            current_price = self.data['close'].iloc[-1]
        else:
            # Fallback to fetching current price from ticker
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
            except Exception as e:
                log_error(e, "fetching current price")
                return
        
        # Check position age and minimum holding time
        min_holding_time_elapsed = True
        if self.position_opened_time is not None:
            if isinstance(self.position_opened_time, datetime):
                holding_time_minutes = (datetime.now() - self.position_opened_time).total_seconds() / 60
                min_holding_time_elapsed = holding_time_minutes >= self.min_holding_time_minutes
                
                if not min_holding_time_elapsed:
                    # Log holding time information periodically
                    if int(time.time()) % 300 < 5:  # Log every ~5 minutes
                        remaining_minutes = self.min_holding_time_minutes - holding_time_minutes
                        logger.info(f"Holding position for {holding_time_minutes:.1f} minutes, " + 
                                f"minimum hold time: {self.min_holding_time_minutes} minutes " +
                                f"({remaining_minutes:.1f} minutes remaining)")
        else:
            # If position_opened_time is not set but we have a position, set it now
            self.position_opened_time = datetime.now()
        
        # Check if we have strategy exit signals in data (only if they exist)
        signal_exit = False
        if hasattr(self, 'data') and self.data is not None:
            if 'exit_long' in self.data.columns and 'exit_short' in self.data.columns:
                # Check exit signals from strategy
                if position_side == 'long' and self.data['exit_long'].iloc[-1]:
                    if min_holding_time_elapsed:
                        logger.info("Exit long signal detected and minimum holding time elapsed")
                        signal_exit = True
                    else:
                        logger.info("Exit long signal detected but minimum holding time not elapsed - holding position")
                elif position_side == 'short' and self.data['exit_short'].iloc[-1]:
                    if min_holding_time_elapsed:
                        logger.info("Exit short signal detected and minimum holding time elapsed")
                        signal_exit = True
                    else:
                        logger.info("Exit short signal detected but minimum holding time not elapsed - holding position")
        
        # Execute exit if signal indicates and minimum time elapsed
        if signal_exit:
            # Cancel existing stop orders before market exit
            if self.exchange_orders.get('stop_loss_id'):
                try:
                    self.exchange.cancel_order(self.exchange_orders['stop_loss_id'], self.symbol)
                except Exception as e:
                    logger.info(f"Error canceling stop loss (may already be executed): {e}")
            
            if self.exchange_orders.get('take_profit_id'):
                try:
                    self.exchange.cancel_order(self.exchange_orders['take_profit_id'], self.symbol)
                except Exception as e:
                    logger.info(f"Error canceling take profit (may already be executed): {e}")
            
            # Execute exit
            logger.info("Executing signal-based exit")
            
            # Try limit chase first for better execution price
            close_side = 'buy' if position_side == 'short' else 'sell'
            
            try:
                # Use limit chase for potentially better execution
                close_order = self.limit_chase_entry(close_side, position_size, max_attempts=2, timeout=3)
                
                if close_order:
                    logger.info(f"Successfully closed position with limit chase on signal: {close_order['id']}")
                    # Record trade with signal reason
                    self._record_exit_trade_with_reason(position, close_order, 'signal')
                    self.trailing_stop_data = None
                    self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                    return
            except Exception as e:
                log_error(e, "limit chase exit on signal")
                # Fall back to market exit
            
            # If limit chase failed or didn't complete, use market exit
            self.market_exit()
            self.trailing_stop_data = None
            self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
            return
        
        # Check for trailing stop hit if it's active and enabled
        if self.trailing_stop_data is not None and current_price is not None and hasattr(self, 'use_trailing_stop') and self.use_trailing_stop:
            if self.check_trailing_stop_hit(self.trailing_stop_data, current_price):
                logger.info(f"Trailing stop hit at {current_price}! Exiting position with market order.")
                self.market_exit()
                self.trailing_stop_data = None
                self.exchange_orders = {'stop_loss_id': None, 'take_profit_id': None}
                return

    def execute_trading_logic(self):
        """Execute the execution logic for the trading strategy."""
        try:
            # Track if we've printed position info this cycle
            position_printed = False
            
            # 1. Check for existing positions (silent check first)
            positions = self.get_open_positions(verbose=False)
            active_position_exists = positions and any(float(pos.get('contracts', 0) or 0) != 0 for pos in positions)
            
            if active_position_exists:
                # Now print position info once (with verbose=True)
                if not position_printed:
                    positions = self.get_open_positions(verbose=True)
                    position_printed = True
                
                logger.info("Active position detected - Managing position")
                
                # NEW: Check if the position was just opened in this session
                just_opened = False
                if hasattr(self, 'position_opened_time') and self.position_opened_time:
                    time_since_opened = (datetime.now() - self.position_opened_time).total_seconds()
                    if time_since_opened < 30:  # Less than 30 seconds old
                        just_opened = True
                        logger.info(f"Position opened very recently ({time_since_opened:.1f}s ago) - letting setup complete")
                
                # NEW: If position was just opened, skip monitoring this cycle to let the setup complete
                if just_opened:
                    return False  # Exit early to let order setup finish
                
                # NEW: Check if we have an unverified market order that now has a position
                if not self.position_verified:
                    logger.info("Found active position that wasn't previously verified - initializing now")
                    
                    # Record the position opening time if not already set
                    if not hasattr(self, 'position_opened_time') or not self.position_opened_time:
                        self.position_opened_time = datetime.now() - timedelta(minutes=1)  # Assume it's been open for at least a minute
                    
                    # Force verification now
                    verification_result = self.verify_and_track_all_orders()
                    logger.info(f"Late position verification result: {verification_result.get('status', 'unknown')}")
                    
                    return False  # Exit early to let everything settle
                
                # CHANGED: Only verify initially - we'll re-verify if needed during monitoring
                if not self.position_verified:
                    # If this is a newly opened position, delay verification
                    if hasattr(self, 'position_opened_time') and self.position_opened_time:
                        time_since_opened = (datetime.now() - self.position_opened_time).total_seconds()
                        
                        # Only delay for positions opened by the bot in this session
                        if time_since_opened < 15:  # 15 second delay for new positions
                            logger.info(f"Position opened {time_since_opened:.1f}s ago - delaying order verification")
                            return False
                    
                    # For existing positions at startup or after delay, verify orders immediately
                    verification_result = self.verify_and_track_all_orders()
                    logger.info(f"Order verification result: {verification_result.get('status', 'unknown')}")
                    
                    # Exit early after first verification to ensure orders are tracked properly
                    return False
                
                # Monitor exit orders - check if intervention is needed
                # The monitor_exit_orders function will now handle re-verification if orders are missing
                intervention_needed = self.monitor_exit_orders()
                if intervention_needed:
                    logger.info("Order monitoring detected need for intervention - market exit executed")
                    return True
                
                # Finally check for exit signals or trailing stop hits
                self.check_exit_conditions()
                
                # Position management continues until exit
                return False
            else:
                # No active position - proceed with signal generation
                logger.info("No active position - Analyzing market for entry signals")
            
            # 3. Generate a new signal using the signal bot
            signal = None
            
            if hasattr(self, 'signal_bot') and self.signal_bot is not None:
                # Use the signal bot if available
                try:
                    # Make sure signal bot has fresh data
                    self.signal_bot.fetch_data()
                    self.signal_bot.calculate_indicators()
                    self.signal_bot.fetch_orderbook()
                    self.signal_bot.fetch_trades()
                    
                    # Generate signal
                    signal = self.signal_bot.run_liquidity_sweep_strategy()
                    
                    # Get current price from signal bot's data
                    if hasattr(self.signal_bot, 'data') and self.signal_bot.data is not None and len(self.signal_bot.data) > 0:
                        self.data = self.signal_bot.data  # Share data with execution bot
                    
                    if signal:
                        logger.info(f"Signal generated: {signal.get('side', 'unknown')} with score {signal.get('sweep_score', 0):.2f}")
                except Exception as e:
                    log_error(e, "getting signal from signal bot")
            
            ### NEW PART ###
            # 4. If we have a valid signal, execute entry
            if signal:
                side = signal.get('side')
                expected_entry_price = signal.get('price')
                
                # Extract SL/TP factors from signal if available
                sl_factor = signal.get('sl_factor')
                tp_factor = signal.get('tp_factor')
                
                # Set the flag to indicate we're starting the position opening process
                self.position_opening_in_progress = True

                # Check whether to use signal-provided SL/TP
                use_signal_sl_tp = (self.use_signal_sl_tp and 
                                'stop_loss' in signal and 
                                'take_profit' in signal)

                if use_signal_sl_tp:
                    stop_loss = signal.get('stop_loss')
                    take_profit = signal.get('take_profit')
                    logger.info(f"Using signal-provided stop loss: {stop_loss} and take profit: {take_profit}")
                else:
                    # Calculate using ExecutionBot's built-in method
                    stop_loss, take_profit = self.calculate_stop_and_target(side, expected_entry_price)
                    
                    # Log appropriate message based on why we're using calculated values
                    if self.use_signal_sl_tp:
                        logger.info(f"Signal SL/TP requested but not provided. Using calculated values.")
                    else:
                        logger.info(f"Using calculated stop loss: {stop_loss} and take profit: {take_profit}")
                                
                # Calculate position size
                position_size = self.calculate_position_size(expected_entry_price, stop_loss)
                
                # Execute entry with market order
                order = self.limit_chase_entry(side, position_size)
                
                if order:
                    # Get actual execution price from the order
                    actual_entry_price = None
                    
                    # Try to get the actual entry price from different possible fields
                    if 'average' in order and order['average']:
                        actual_entry_price = float(order['average'])
                    elif 'price' in order and order['price']:
                        actual_entry_price = float(order['price'])
                    
                    # If we couldn't get a price from the order, fetch position details
                    if not actual_entry_price:
                        try:
                            # Wait a moment for the exchange to process the order
                            time.sleep(1)
                            positions = self.get_open_positions(verbose=False)
                            if positions and len(positions) > 0:
                                position = positions[0]
                                actual_entry_price = float(position.get('entryPrice', expected_entry_price))
                        except Exception as e:
                            logger.warning(f"Could not fetch actual entry price from position: {e}")
                            actual_entry_price = expected_entry_price
                    
                    # If we still don't have an actual price, use the expected price
                    if not actual_entry_price:
                        logger.warning("Could not determine actual entry price, using expected price")
                        actual_entry_price = expected_entry_price
                    
                    logger.info(f"Position opened: {side.upper()} {position_size} contracts at {actual_entry_price}")
                    logger.info(f"Expected price: {expected_entry_price}, Actual price: {actual_entry_price}")
                    
                    
                    if 'stop_loss' not in signal or 'take_profit' not in signal or not self.use_signal_sl_tp:
                        # Recalculate stop loss and take profit based on actual entry price
                        if sl_factor is not None and tp_factor is not None:
                            # Use the same factors from the signal to recalculate
                            if self.use_atr_for_stops and hasattr(self, 'atr'):
                                # Use ATR-based calculation with factors
                                atr = self.atr
                                if side == 'buy':
                                    stop_loss = actual_entry_price - (atr * sl_factor)
                                    take_profit = actual_entry_price + (atr * tp_factor)
                                else:  # sell
                                    stop_loss = actual_entry_price + (atr * sl_factor)
                                    take_profit = actual_entry_price - (atr * tp_factor)
                                logger.info(f"Recalculated SL/TP using signal factors with ATR: SL={stop_loss}, TP={take_profit}")
                            else:
                                # Use percentage-based calculation with factors
                                base_sl_percent = self.base_sl_percent if hasattr(self, 'base_sl_percent') else 1.0
                                base_tp_percent = self.base_tp_percent if hasattr(self, 'base_tp_percent') else 1.0
                                
                                sl_percent = base_sl_percent * sl_factor
                                tp_percent = base_tp_percent * tp_factor
                                
                                if side == 'buy':
                                    stop_loss = actual_entry_price * (1 - sl_percent/100)
                                    take_profit = actual_entry_price * (1 + tp_percent/100)
                                else:  # sell
                                    stop_loss = actual_entry_price * (1 + sl_percent/100)
                                    take_profit = actual_entry_price * (1 - tp_percent/100)
                                logger.info(f"Recalculated SL/TP using signal factors with percentages: SL={stop_loss}, TP={take_profit}")
                        else:
                            # Fallback to standard calculation if no factors available
                            stop_loss, take_profit = self.calculate_stop_and_target(side, actual_entry_price)
                            logger.info(f"Recalculated SL/TP based on actual entry: SL={stop_loss}, TP={take_profit}")
                    
                    # Set exchange stop orders with actual entry price
                    self.exchange_orders = self.set_exchange_stop_orders(
                        side, actual_entry_price, stop_loss, take_profit, position_size)
                    
                    # Initialize tiered trailing stop with actual entry price
                    if self.use_trailing_stop:
                        self.trailing_stop_data = self.initialize_tiered_trailing_stop(
                            actual_entry_price, stop_loss, take_profit, 'long' if side == 'buy' else 'short')
                    else:
                        self.trailing_stop_data = None
                        logger.info("Trailing stop disabled by configuration")
                    
                    # Record position opening time
                    self.position_opened_time = datetime.now()
                    self.position_verified = False
                    
                    # ADD THIS: Delay before first verification to allow orders to appear on exchange
                    logger.info(f"Waiting 10 seconds before initial order verification...")
                    time.sleep(10)  # 10-second delay

                    # ADD THIS: Now run initial verification to check orders were placed correctly
                    verification_result = self.verify_and_track_all_orders()
                    logger.info(f"Initial order verification result: {verification_result.get('status', 'unknown')}")

                    # IMPORTANT: Only now clear the position opening flag - this will allow monitoring to start
                    self.position_opening_in_progress = False

                    return True
                else:
                    logger.error("Order execution failed or verification failed. No action taken.")
                    return False
            
            return False
        
        except Exception as e:
            log_error(e, "execution loop")
            return False
 
    def start_trading_loop(self, interval=60, max_cycles=None):
        """Start the trading loop and the position monitor"""
        # Start the position monitor first
        self.start_position_monitor()
        logger.info(f"Starting execution trading loop with {interval} second interval")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"Starting trading cycle {cycle_count}")
                
                try:
                    # Execute one cycle of trading logic
                    self.execute_trading_logic()
                    
                    # Check if we've reached max cycles
                    if max_cycles and cycle_count >= max_cycles:
                        logger.info(f"Reached maximum cycle count of {max_cycles}")
                        break
                    
                    # Sleep until next cycle
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    logger.info("Trading loop interrupted by user")
                    break
                except Exception as e:
                    log_error(e, f"trading cycle {cycle_count}")
                    time.sleep(30)  # Wait longer before retrying on error
            
            logger.info("Trading loop terminated")
        
        except Exception as e:
            log_error(e, "trading loop")
    

