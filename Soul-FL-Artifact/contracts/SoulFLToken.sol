// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

/// @notice Adapter to a deployed enrollment proof verifier. The repository does
/// not supply a Groth16 circuit or generated verifier; empty proofs cannot enroll.
interface IEnrollmentVerifier {
    function verifyEnrollment(address key, bytes calldata anchor, bytes calldata proof) external view returns (bool);
}

/// @notice Enrollment state and coordinator-signed, once-per-key-per-round credits.
/// The contract authorizes vouchers but cannot check off-chain reconstruction scores.
contract SoulFLToken {
    uint256 public constant SCALE = 1_000_000;
    uint256 public constant DECAY_FACTOR = 951_229;
    uint256 public constant INITIAL_BALANCE = 100 * SCALE;
    uint256 public constant CREDIT = 10 * SCALE;
    uint256 public constant ELIGIBILITY_GATE = 2 * SCALE;
    uint256 private constant SECP256K1_HALF_N =
        0x7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0;
    bytes32 private constant DOMAIN_TYPEHASH = keccak256('EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)');
    bytes32 private constant VOUCHER_TYPEHASH = keccak256('Voucher(address key,bytes32 commitment,uint256 round,uint256 credit,uint256 nonce)');
    bytes32 public immutable DOMAIN_SEPARATOR;
    address public immutable coordinator;
    IEnrollmentVerifier public immutable verifier;
    uint256 public currentRound = 1;

    struct ClientState {
        bytes32 anchorCommitment;
        uint256 balance;
        uint256 lastRound;
        uint256 nonce;
        uint256 lastCreditRound;
        bool enrolled;
    }
    mapping(address => ClientState) public clients;
    event Enrolled(address indexed key, bytes32 indexed commitment);
    event Credited(address indexed key, uint256 indexed round, uint256 nonce, uint256 newBalance);
    event RoundAdvanced(uint256 indexed round);

    modifier onlyCoordinator() {
        require(msg.sender == coordinator, 'coordinator only');
        _;
    }

    constructor(address verifierAddress) {
        require(verifierAddress.code.length > 0, 'real verifier contract required');
        coordinator = msg.sender;
        verifier = IEnrollmentVerifier(verifierAddress);
        DOMAIN_SEPARATOR = keccak256(abi.encode(DOMAIN_TYPEHASH, keccak256('SoulFLVoucher'),
            keccak256('1'), block.chainid, address(this)));
    }

    function enroll(bytes calldata anchor, bytes calldata proof) external {
        require(!clients[msg.sender].enrolled, 'key already enrolled');
        require(anchor.length > 0 && proof.length > 0, 'anchor and proof required');
        require(verifier.verifyEnrollment(msg.sender, anchor, proof), 'invalid enrollment proof');
        bytes32 commitment = keccak256(abi.encode(msg.sender, anchor));
        clients[msg.sender] = ClientState(commitment, INITIAL_BALANCE, currentRound, 0, 0, true);
        emit Enrolled(msg.sender, commitment);
    }

    function effectiveBalance(address key) public view returns (uint256) {
        ClientState storage account = clients[key];
        if (!account.enrolled) return 0;
        uint256 result = account.balance;
        for (uint256 j = account.lastRound; j < currentRound; ++j) {
            result = result * DECAY_FACTOR / SCALE;
        }
        return result;
    }

    function claim(uint256 round, uint256 nonce, bytes calldata signature) external {
        ClientState storage account = clients[msg.sender];
        require(account.enrolled, 'not enrolled');
        require(round == currentRound, 'wrong round');
        require(nonce == account.nonce, 'wrong nonce');
        require(account.lastCreditRound < round, 'already credited this round');
        uint256 balance = effectiveBalance(msg.sender);
        require(balance > ELIGIBILITY_GATE, 'key excluded');
        bytes32 structHash = keccak256(abi.encode(VOUCHER_TYPEHASH, msg.sender,
            account.anchorCommitment, round, CREDIT, nonce));
        bytes32 digest = keccak256(abi.encodePacked(hex'1901', DOMAIN_SEPARATOR, structHash));
        require(_recover(digest, signature) == coordinator, 'invalid voucher');
        account.nonce = nonce + 1;
        account.lastCreditRound = round;
        account.balance = balance + CREDIT;
        account.lastRound = currentRound;
        emit Credited(msg.sender, currentRound, nonce, account.balance);
    }

    function advanceRound() external onlyCoordinator {
        ++currentRound;
        emit RoundAdvanced(currentRound);
    }

    function _recover(bytes32 digest, bytes calldata signature) private pure returns (address) {
        require(signature.length == 65, 'bad signature length');
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := calldataload(signature.offset)
            s := calldataload(add(signature.offset, 32))
            v := byte(0, calldataload(add(signature.offset, 64)))
        }
        require(uint256(s) <= SECP256K1_HALF_N && (v == 27 || v == 28), 'noncanonical signature');
        address signer = ecrecover(digest, v, r, s);
        require(signer != address(0), 'invalid signer');
        return signer;
    }
}
