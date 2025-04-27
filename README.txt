# Crypto Trading Bot

An automated cryptocurrency trading system built for Hyperliquid Exchange.

## Features

- Automated technical analysis using pandas_ta
- Advanced orderbook analysis
- Dynamic regime detection and parameter adjustment
- Real-time trade execution
- Risk management and position sizing

## Setup Instructions

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your credentials
4. Run the bot: `python Tradingbot.py`

## Configuration

The bot can be configured through environment variables in the `.env` file:

- `wallet`: Your Hyperliquid wallet address
- `key`: Your Hyperliquid private key

- To adjust settings like stop-loss and take-profit levels, orderbook or technical indicator weights, minimum signal sweep score or quality, go into SignalBotV7.py and change parameters in the initialise_regime_parameters function and in Tradingbot adjust the weights. 
- To adjust trailing-stop settings, go into ExecutionBotV7.py and adjust numbers in the initialise_trailing_stop method. You can also turn of the use of trailing stop entirely in the Tradingbot.py by setten the use_trailing_stop to False

## Disclaimer

This tradingbot in no way guarantees profit. You may lose all your money. Use at your own risk.
Trading cryptocurrency involves substantial risk of loss and is not suitable for everyone.
