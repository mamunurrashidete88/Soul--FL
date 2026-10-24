// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;
import './SoulFLToken.sol';
/// @notice Negative-test verifier: it NEVER validates an enrollment proof.
contract RejectingVerifier is IEnrollmentVerifier {
    function verifyEnrollment(address, bytes calldata, bytes calldata) external pure returns (bool) {
        return false;
    }
}
