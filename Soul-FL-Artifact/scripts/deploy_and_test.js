// Supply a deployed Groth16 verifier adapter. No proof is fabricated here.
const hre = require('hardhat');
async function main() {
  const verifier = process.env.ENROLLMENT_VERIFIER;
  if (!verifier || (await hre.ethers.provider.getCode(verifier)) === '0x') {
    throw new Error('Set ENROLLMENT_VERIFIER to a deployed, real enrollment verifier');
  }
  const Contract = await hre.ethers.getContractFactory('SoulFLToken');
  const contract = await Contract.deploy(verifier);
  await contract.waitForDeployment();
  console.log('SoulFLToken:', await contract.getAddress());
  console.log('Enrollment verifier:', verifier);
  console.log('Round:', (await contract.currentRound()).toString());
}
main().catch(error => { console.error(error); process.exitCode = 1; });
