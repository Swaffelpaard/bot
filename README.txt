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

## Disclaimer

This trading bot is provided for educational purposes only. Use at your own risk.
Trading cryptocurrency involves substantial risk of loss and is not suitable for everyone.
