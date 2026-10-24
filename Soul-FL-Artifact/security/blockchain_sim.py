
from dataclasses import dataclass
import json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature
from security.trust_engine import SCALE, effective_balance, eligible

@dataclass
class Account:
    commitment: str
    balance: int
    last_round: int
    nonce: int = 0
    last_credit_round: int = 0

class VoucherLedger:
    def __init__(self, signer=None, domain='soul-fl-local-v1', initial=100.0, credit=10.0, gate=2.0):
        self.signer = signer or Ed25519PrivateKey.generate()
        self.public_key = self.signer.public_key()
        self.domain = domain
        self.initial = round(initial * SCALE)
        self.credit = round(credit * SCALE)
        self.gate = gate
        self.accounts = {}
        self.round = 1

    def enroll(self, key: str, commitment: str, relation_checked=False):
        if not relation_checked or key in self.accounts or not commitment:
            raise ValueError('Enrollment relation must be checked and key must be new')
        self.accounts[key] = Account(commitment, self.initial, self.round)

    def enroll_assumed(self, key: str, commitment: str):

        if key in self.accounts or not commitment:
            raise ValueError('Key must be new and commitment nonempty')
        self.accounts[key] = Account(commitment, self.initial, self.round)

    def balance(self, key: str) -> int:
        account = self.accounts[key]
        return effective_balance(account.balance, account.last_round, self.round)

    def is_eligible(self, key: str) -> bool:
        return eligible(self.balance(key), self.gate)

    def voucher(self, key: str):
        account = self.accounts[key]
        fields = {'key': key, 'commitment': account.commitment, 'round': self.round,
                  'credit': self.credit, 'nonce': account.nonce, 'domain': self.domain}
        body = json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()
        return fields, self.signer.sign(body)

    def apply(self, fields, signature):
        key = fields['key']
        account = self.accounts[key]
        body = json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()
        try:
            self.public_key.verify(signature, body)
        except InvalidSignature as exc:
            raise ValueError('Invalid coordinator signature') from exc
        if fields != {'key': key, 'commitment': account.commitment, 'round': self.round,
                      'credit': self.credit, 'nonce': account.nonce, 'domain': self.domain}:
            raise ValueError('Wrong domain, round, amount, anchor, or nonce')
        if not self.is_eligible(key):
            raise ValueError('Excluded keys cannot earn credit')
        if account.last_credit_round == self.round:
            raise ValueError('Only one credit per key and round')
        account.balance = self.balance(key) + self.credit
        account.last_round = self.round
        account.last_credit_round = self.round
        account.nonce += 1

    def advance(self):
        self.round += 1
