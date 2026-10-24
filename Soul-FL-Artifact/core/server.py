
from collections import deque
import numpy as np
import torch
from security.projector import HeadProjector
from security.cvae import ConditionalVAE
from core.aggregation import weighted_update

class Server:
    def __init__(self, model, anchor_dim, device='cpu', components=1024,
                 hidden=512, latent=128, kl=0.1, detector_lr=0.001, window=5,
                 snapshots_per_submission=11):
        self.model = model.to(device)
        self.device = device
        self.projector = HeadProjector(components)
        self.detector = ConditionalVAE(components, anchor_dim, latent, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=detector_lr)
        self.kl = kl
        self.window = window
        self.snapshots_per_submission = snapshots_per_submission
        self.anchors = {}
        self.calibration_vectors = []
        self.calibration_finals = []
        self.accepted_scores = deque(maxlen=window)
        self.last_cutoff = None
        self.pending = []
        self.cutoff = None

    def register_anchor(self, key, anchor):
        self.anchors[key] = np.asarray(anchor, dtype=np.float32)

    def _batch(self, head_vectors, anchors):
        projected = self.projector.transform(head_vectors)
        x = torch.from_numpy(projected).to(self.device)
        c = torch.from_numpy(np.asarray(anchors, dtype=np.float32)).to(self.device)

        c = torch.asinh(c / 1000.0)
        return x, c

    def _train_detector(self, x, c, epochs=1):
        self.detector.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            loss = self.detector.loss(x, c, self.kl)
            loss.backward()
            self.optimizer.step()

    def start_round(self, round_number):
        self.pending = []
        if round_number > 10:
            recent = [score for block in self.accepted_scores for score in block]
            self.cutoff = float(np.quantile(recent, 0.95)) if recent else self.last_cutoff
            if self.cutoff is None:
                raise RuntimeError('Detector has no calibrated cutoff')
            self.last_cutoff = self.cutoff

    def screen(self, key, head_vector, snapshots, round_number):
        if key not in self.anchors:
            raise ValueError('Unenrolled key')
        if round_number <= 10:
            if len(snapshots) != self.snapshots_per_submission:
                raise ValueError('Unexpected number of calibration snapshots')
            self.calibration_vectors.extend(snapshots + [head_vector])
            self.calibration_finals.append((head_vector, self.anchors[key]))
            return True, None
        x, c = self._batch([head_vector], [self.anchors[key]])
        score = float(self.detector.score(x, c).item())
        if score <= self.cutoff:
            self.pending.append((head_vector, self.anchors[key], score))
            return True, score
        return False, score

    def aggregate(self, deltas, balances):
        return weighted_update(self.model, deltas, balances)

    def finish_round(self, round_number):
        if round_number == 10:
            self.projector.fit(self.calibration_vectors)
            heads, anchors = zip(*self.calibration_finals)
            x, c = self._batch(heads, anchors)
            self._train_detector(x, c, epochs=20)
            scores = self.detector.score(x, c).cpu().numpy()
            self.last_cutoff = float(np.quantile(scores, 0.95))
        elif round_number > 10:
            self.accepted_scores.append([entry[2] for entry in self.pending])
            if self.pending:
                heads, anchors, _ = zip(*self.pending)
                x, c = self._batch(heads, anchors)
                self._train_detector(x, c)

    @torch.no_grad()
    def evaluate(self, loader, source=2, target=8):
        self.model.eval()
        correct = source_total = source_target = count = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            predicted = self.model(inputs).argmax(1)
            correct += int((predicted == labels).sum())
            count += len(labels)
            mask = (labels == source)
            source_total += int(mask.sum())
            source_target += int((predicted[mask] == target).sum())
        return 100 * correct / count, 100 * source_target / source_total if source_total else float('nan')
