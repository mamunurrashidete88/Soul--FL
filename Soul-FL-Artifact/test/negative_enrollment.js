const assert = require('node:assert/strict');
const hre = require('hardhat');
describe('SoulFLToken negative enrollment', function () {
  it('requires a verifier contract and rejects invalid enrollment', async function () {
    const [coordinator, client] = await hre.ethers.getSigners();
    const Rejecting = await hre.ethers.getContractFactory('RejectingVerifier');
    const rejecting = await Rejecting.deploy();
    await rejecting.waitForDeployment();
    const SoulFL = await hre.ethers.getContractFactory('SoulFLToken');
    const soul = await SoulFL.deploy(await rejecting.getAddress());
    await soul.waitForDeployment();
    await assert.rejects(soul.connect(client).enroll('0x1234', '0x1234'), /invalid enrollment proof/);
    assert.equal(await soul.effectiveBalance(client.address), 0n);
    await soul.connect(coordinator).advanceRound();
    assert.equal(await soul.currentRound(), 2n);
    await assert.rejects(soul.connect(client).claim(2, 0, '0x'), /not enrolled/);
  });
});
