# test.py
from web3 import Web3
print("✅ web3 imported successfully!")
w3 = Web3(Web3.HTTPProvider("https://rpc.primordial.bdagscan.com"))
print("Connected:", w3.is_connected())