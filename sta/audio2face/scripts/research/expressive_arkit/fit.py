"""Experimental 52-channel retargeting of the complete NVIDIA prediction.

43 face channels: bounded geometry solve with lip-distance observations.
8 gaze channels: signed eye rotations, using a documented canonical 25-degree
endpoint (this is our calibration, not an Apple or NVIDIA calibration).
tongueOut: tongueStretch coefficient after removing authoring jaw/mouth motion.
The latter is an explicit protrusion proxy, not a measured ARKit capture label.
"""
import json
import sys
from pathlib import Path
import numpy as np

EYE_ENDPOINT_DEGREES = 25.
PAIRS = np.array([[9013,10613],[941,10433],[8859,10434],[2963,3726],
                  [1439,1475],[5980,7505],[6093,18810],[263,18809],[17362,13358]])


def mm(a, b):
    # Explicit contraction avoids NumPy/Accelerate's spurious matmul exception
    # flags on this host; all inputs and outputs are still checked for finiteness.
    return np.einsum('ik,kj->ij', a, b, optimize=False)


def bounded_solve(rhs, gram, iterations=160):
    inverse = np.linalg.solve(gram, np.eye(len(gram), dtype=gram.dtype))
    current = np.clip(mm(rhs, inverse), 0, 1)
    extrapolated = current.copy()
    step = np.float32(1 / np.linalg.eigvalsh(gram)[-1])
    momentum = 1.
    for _ in range(iterations):
        updated = np.clip(extrapolated - step * (mm(extrapolated, gram) - rhs), 0, 1)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = updated + np.float32((momentum - 1) / next_momentum) * (updated - current)
        current, momentum = updated, next_momentum
    if not np.isfinite(current).all():
        raise FloatingPointError('Nonfinite bounded geometry fit')
    return current


def gaze_coefficients(eyes):
    """NVIDIA order: right pitch/yaw, left pitch/yaw, local rig degrees."""
    eyes = np.asarray(eyes, np.float32)
    if eyes.ndim != 2 or eyes.shape[1] != 4 or not np.isfinite(eyes).all():
        raise ValueError('Expected finite [frames,4] eye rotations')
    result = np.zeros((len(eyes), 52), np.float32)
    # Positive local X looks down. Inward Y has opposite signs on each side.
    for pitch, yaw, down, inside, outside, up, inward_sign in (
        (2,3,1,2,3,4,-1), (0,1,8,9,10,11,1)
    ):
        result[:,down] = np.maximum(eyes[:,pitch], 0)
        result[:,up] = np.maximum(-eyes[:,pitch], 0)
        result[:,inside] = np.maximum(eyes[:,yaw] * inward_sign, 0)
        result[:,outside] = np.maximum(-eyes[:,yaw] * inward_sign, 0)
    return np.clip(result / EYE_ENDPOINT_DEGREES, 0, 1)


def safe_offset(nodes, parent, positions):
    from prepare_rig import attribute
    xyz = np.column_stack((positions, np.ones(len(positions))))
    while parent:
        _, parent, body = nodes[parent]
        matrix = attribute(body, 'opm', 'matrix', np.eye(4).ravel()).reshape(4,4)
        xyz = mm(xyz, matrix)
    if not np.isfinite(xyz).all():
        raise FloatingPointError('Invalid Maya transform')
    return xyz[:,:3]


class FullFaceFit:
    def __init__(self, assets, scene, app):
        sys.path.insert(0, str(Path(app) / 'scripts'))
        import prepare_blendshapes as pack
        from prepare_rig import maya_nodes
        # Only this research process's function binding changes; no source edits.
        pack.offset_transform = safe_offset
        with np.load(assets / 'bs_skin_James.npz') as rig, np.load(
            assets / 'model_data_James.npz'
        ) as model, np.load(assets / 'bs_tongue_James.npz') as tongue:
            self.names = [n.decode() for n in rig['poseNames'][1:]]
            self.eye_ids = np.array([1,2,3,4,8,9,10,11])
            self.face_ids = np.array([i for i in range(51) if i not in self.eye_ids])
            self.shape = np.stack([rig[n] for n in self.names])
            self.neutral = rig['neutral'].copy()
            self.model_neutral = model['neutral_skin'].copy()
            self.mask = rig['frontalMask'].copy()
            self.model_tongue = model['neutral_tongue'].copy()
            self.tongue_neutral = tongue['neutral'].copy()
            self.tongue_names = [n.decode() for n in tongue['poseNames'][1:]]
            self.tongue_basis = np.stack([tongue[n].ravel() for n in self.tongue_names])
        if len(self.names) != 52 or len(self.tongue_names) != 16:
            raise ValueError('Unexpected James source rig schema')
        self.upper, self.lower = PAIRS.T
        self.gap_basis = self.shape[:,self.upper,1] - self.shape[:,self.lower,1]
        self.gap_neutral = self.neutral[self.upper,1] - self.neutral[self.lower,1]
        front = self.model_neutral[self.mask]
        mouth = (np.abs(front[:,0]) < 3) & (front[:,1] > 166.1) & (front[:,1] < 169) & (front[:,2] > 5)
        weight = np.repeat(np.where(mouth,4.,1.).astype(np.float32),3)
        self.basis = self.shape[self.face_ids][:,self.mask].reshape(43,-1)
        self.weighted_basis = self.basis * weight
        lip = self.gap_basis[self.face_ids]
        gram = mm(self.basis, self.weighted_basis.T) + 192 * mm(lip, lip.T)
        # A fixed ridge of 1 suppressed weak eyelid targets by ~30% even on
        # exact synthetic poses. Scale regularization by each column's energy.
        self.gram = gram + np.diag(np.diag(gram)*np.float32(.0001))
        self.tongue_gram = mm(self.tongue_basis, self.tongue_basis.T) + np.eye(16,dtype=np.float32)*.1
        nodes = maya_nodes(Path(scene).read_text())
        neutral, deltas = pack.maya_deltas(nodes,'c_tongue_hi_bs','c_tongue_hiShapeOrig', self.names)
        self.tongue_mouth = np.stack([deltas.get(n,np.zeros_like(neutral)) for n in self.names])
        self.tongue_out_index = self.tongue_names.index('tongueStretch')

    def __call__(self, geometry):
        geometry = np.asarray(geometry, np.float32)
        if geometry.ndim != 2 or geometry.shape[1] != 88831 or not np.isfinite(geometry).all():
            raise ValueError('Expected finite complete NVIDIA geometry [frames,88831]')
        eyes = geometry[:,-4:]
        result = gaze_coefficients(eyes)
        skin = geometry[:,:72006].reshape(-1,24002,3) + self.model_neutral
        target = skin - self.neutral - np.einsum('fc,cvk->fvk',result,self.shape,optimize=False)
        rhs = mm(target[:,self.mask].reshape(len(result),-1), self.weighted_basis.T)
        gap = target[:,self.upper,1] - target[:,self.lower,1]
        rhs += 192 * mm(gap,self.gap_basis[self.face_ids].T)
        result[:,self.face_ids] = bounded_solve(rhs,self.gram)
        raw_tongue = geometry[:,72006:88812].reshape(-1,5602,3) + self.model_tongue
        tongue_target = raw_tongue - self.tongue_neutral - np.einsum(
            'fc,cvk->fvk',result,self.tongue_mouth,optimize=False)
        tongue_weights = bounded_solve(mm(tongue_target.reshape(len(result),-1),self.tongue_basis.T), self.tongue_gram)
        result[:,51] = tongue_weights[:,self.tongue_out_index]
        reconstructed = np.einsum('fc,cvk->fvk',result,self.shape,optimize=False) + self.neutral
        residual = np.sqrt(np.mean((reconstructed[:,self.mask]-skin[:,self.mask])**2,axis=(1,2)))
        return result, dict(eyes=eyes.copy(), tongue_weights=tongue_weights,
                           jaw=geometry[:,88812:88827].copy(), skin_fit_rms=residual)

    def save_calibration(self, destination):
        np.savez_compressed(destination, shape=self.shape, neutral=self.neutral,
                            gap_basis=self.gap_basis, gap_neutral=self.gap_neutral,
                            tongue_mouth=self.tongue_mouth,
                            tongue_out=self.tongue_basis[self.tongue_out_index].reshape(-1,3),
                            eye_endpoint_degrees=np.array(EYE_ENDPOINT_DEGREES))
