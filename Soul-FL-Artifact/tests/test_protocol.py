
import unittest
import numpy as np
from security.zk import image_histogram, perturb, anchor_commitment, verify_image_relation
from security.blockchain_sim import VoucherLedger
from security.trust_engine import SCALE, aggregation_weights
from security.projector import HeadProjector
from data_loader import dirichlet_partition

class EnrollmentTests(unittest.TestCase):
    def test_record_histogram_and_binding(self):
        images = np.full((64, 1, 28, 28), 8, dtype=np.uint8)
        hist = image_histogram(images)
        self.assertEqual(hist.shape, (64,))
        self.assertEqual(hist[2], 64 * 28 * 28)
        anchor, noise = perturb(hist, np.random.default_rng(3))
        c = anchor_commitment('key-a', anchor)
        self.assertTrue(verify_image_relation(images, anchor, noise, 'key-a', c))
        self.assertFalse(verify_image_relation(images[:63], anchor, noise, 'key-a', c))
        self.assertFalse(verify_image_relation(images, anchor, noise, 'key-b', c))
        modified = anchor.copy()
        modified[2] += 1
        self.assertFalse(verify_image_relation(images, modified, noise, 'key-a', c))
        # The same records can validly enroll with another key and new noise.
        a2, n2 = perturb(hist, np.random.default_rng(4))
        self.assertTrue(verify_image_relation(images, a2, n2, 'key-b', anchor_commitment('key-b', a2)))

    def test_dirichlet_assignment_uses_each_record_once(self):
        labels = np.tile(np.arange(10), 100)
        split = dirichlet_partition(labels, clients=5, alpha=0.5, min_size=10, seed=11)
        combined = sorted(v for part in split for v in part)
        self.assertEqual(combined, list(range(len(labels))))

class VoucherTests(unittest.TestCase):
    def test_precredit_weight_domain_and_replay(self):
        ledger = VoucherLedger()
        ledger.enroll('honest', 'anchor-h', relation_checked=True)
        ledger.enroll('attacker', 'anchor-a', relation_checked=True)
        before = aggregation_weights([ledger.balance('honest'), ledger.balance('attacker')])
        self.assertEqual(before, [0.5, 0.5])
        payload, signature = ledger.voucher('honest')
        ledger.apply(payload, signature)
        with self.assertRaises(ValueError):
            ledger.apply(payload, signature)
        fresh, fresh_signature = ledger.voucher('honest')
        with self.assertRaisesRegex(ValueError, 'Only one credit'):
            ledger.apply(fresh, fresh_signature)
        ledger.advance()
        self.assertGreater(ledger.balance('honest'), ledger.balance('attacker'))
        forged, _ = ledger.voucher('attacker')
        forged = dict(forged)
        forged['domain'] = 'different'
        with self.assertRaises(ValueError):
            ledger.apply(forged, signature)

    def test_absorbing_gate(self):
        ledger = VoucherLedger()
        ledger.enroll('key', 'anchor', relation_checked=True)
        for _ in range(90):
            ledger.advance()
        self.assertFalse(ledger.is_eligible('key'))
        voucher, signature = ledger.voucher('key')
        with self.assertRaises(ValueError):
            ledger.apply(voucher, signature)

class ProjectorTests(unittest.TestCase):
    def test_fit_uses_calibration_vectors(self):
        vectors = np.random.default_rng(12).normal(size=(14, 5)).astype(np.float32)
        pca = HeadProjector(components=2).fit(vectors)
        self.assertEqual(pca.fitted_on, 14)
        self.assertEqual(pca.transform(vectors[:1]).shape, (1, 2))

if __name__ == '__main__':
    unittest.main()
