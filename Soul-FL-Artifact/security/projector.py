import numpy as np
from sklearn.decomposition import IncrementalPCA

class HeadProjector:
    def __init__(self, components=1024):
        self.components = components
        self.pca = None
        self.fitted_on = 0

    def fit(self, vectors):
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] <= self.components or x.shape[1] < self.components:
            raise ValueError('Need more calibration vectors than components and enough head coordinates')
        self.pca = IncrementalPCA(n_components=self.components, batch_size=max(self.components, 2 * self.components))
        self.pca.fit(x)
        self.fitted_on = len(x)
        return self

    def transform(self, vectors):
        if self.pca is None:
            raise RuntimeError('Projector is not calibrated')
        x = np.asarray(vectors, dtype=np.float32)
        return self.pca.transform(np.atleast_2d(x)).astype(np.float32)
