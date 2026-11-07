"""Head-relative geometry measurements shared by native and fitted motion."""
import json
import struct
import numpy as np

LIP_IDS = np.array([[9013, 10613], [941, 10433], [8859, 10434],
                    [2963, 3726], [1439, 1475], [5980, 7505],
                    [6093, 18810], [263, 18809], [17362, 13358]]).ravel()
NAMES = ['mouthNarrowing', 'lipProtrusion', 'jawOpening', 'lowerLipIncisorProximity']


class MotionFeatures:
    def __init__(self, rig_dir, assets):
        with np.load(rig_dir / 'model_data_James.npz') as model:
            self.native_skin = model['neutral_skin'][LIP_IDS].copy()
        with np.load(rig_dir / 'bs_skin_James.npz') as rig:
            self.channels = [name.decode() for name in rig['poseNames'][1:]]
            self.fit_skin = rig['neutral'][LIP_IDS].copy()
            self.skin_basis = np.stack([rig[name][LIP_IDS] for name in self.channels]).reshape(52, -1)
        self.neutral_width, self.neutral_protrusion = self.lip_shape(self.native_skin[None])
        manifest = json.loads((assets / 'rig.json').read_text())
        binary = (assets / 'rig.bin').read_bytes()
        teeth = manifest['meshes']['upperTeeth']
        vertices = np.frombuffer(binary, dtype='<f4', count=teeth['vertexCount'] * 3, offset=teeth['positionOffset']).reshape(-1, 3)
        keep = (np.abs(vertices[:, 0]) < 1.3) & (vertices[:, 1] < 168.6) & (vertices[:, 2] > 4.5)
        self.origin = np.array([0, 168, 6], dtype=np.float32)
        self.incisors = np.unique(vertices[keep], axis=0).copy() - self.origin
        if len(self.incisors) < 20:
            raise ValueError('Upper incisor patch is missing or in incompatible coordinates')
        self.inc_norm = np.sum(self.incisors ** 2, axis=1)
        self.jaw_basis = self.read_jaw_basis((assets / 'james-blendshapes.bin').read_bytes())

    @staticmethod
    def read_jaw_basis(binary):
        if binary[:4] != b'JBS1':
            raise ValueError('Invalid James blendshape data')
        offset = 4
        jaw = np.zeros((53, 5, 3), dtype=np.float32)
        for group in range(3):
            count, = struct.unpack_from('<I', binary, offset)
            offset += 4
            for _ in range(count):
                channel, vertices, scale = struct.unpack_from('<IIf', binary, offset)
                offset += 12
                points = np.frombuffer(binary, dtype='<i2', count=vertices * 4, offset=offset).reshape(-1, 4)
                if group == 2:
                    jaw[channel, points[:, 0]] = points[:, 1:] * scale
                offset += points.nbytes
        if offset != len(binary):
            raise ValueError('Unexpected blendshape binary length')
        return jaw

    @staticmethod
    def lip_shape(points):
        left = points[:, :2].mean(axis=1)
        right = points[:, -2:].mean(axis=1)
        width = np.linalg.norm(right - left, axis=1)
        # Head coordinates are fixed in native speech geometry. Moving mouth
        # corners are unsuitable as a reference: they advance with pucker.
        protrusion = points[:, 6:12, 2].mean(axis=1)
        return width, protrusion

    def measure(self, points, jaw_delta):
        width, protrusion = self.lip_shape(points)
        lower_lip = points[:, [7, 9, 11]].reshape(-1, 3) - self.origin
        # Exact nearest-vertex distance to the dense static incisor patch.
        # This is a proximity proxy, not a signed surface-contact detector.
        square = np.sum(lower_lip ** 2, axis=1)[:, None] + self.inc_norm[None] - 2 * (lower_lip @ self.incisors.T)
        proximity = -np.sqrt(np.maximum(0, square.min(axis=1))).reshape(len(points), 3).min(axis=1)
        return np.stack([self.neutral_width - width, protrusion - self.neutral_protrusion,
                         -jaw_delta[:, :, 1].mean(axis=1), proximity], axis=1).astype('<f4')

    def native(self, geometry):
        points = geometry[:, :72006].reshape(-1, 24002, 3)[:, LIP_IDS] + self.native_skin
        jaw = geometry[:, 88812:88827].reshape(-1, 5, 3)
        return self.measure(points, jaw)

    def fitted(self, coefficients):
        points = (coefficients @ self.skin_basis).reshape(-1, len(LIP_IDS), 3) + self.fit_skin
        jaw = (coefficients @ self.jaw_basis[:52].reshape(52, -1)).reshape(-1, 5, 3) + self.jaw_basis[52]
        return self.measure(points, jaw)
