from eth_account import Account
from eth_account.messages import encode_defunct
import hashlib
import json

class EIP712Signer:
    def __init__(self, private_key: str, chain_id: int):
        self.account = Account.from_key(private_key)
        self.chain_id = chain_id

    def sign_rebalance_signal(self, signal: dict) -> dict:
        """Sign rebalance signal (simplified version)"""
        message_data = {
            "fromStrategy": signal["fromStrategy"],
            "toStrategy": signal["toStrategy"],
            "amount": signal["amount"],
            "timestamp": signal["timestamp"]
        }

        # Create message hash
        message_str = json.dumps(message_data, sort_keys=True)
        message_hash = hashlib.sha256(message_str.encode()).hexdigest()

        # Correct signing
        msg = encode_defunct(hexstr=f"0x{message_hash}")
        signed = self.account.sign_message(msg)

        signal["signature"] = signed.signature.hex()
        signal["messageHash"] = f"0x{message_hash}"

        return signal
