"""Derive a smooth contact capture interval from the frozen viseme library."""
import numpy as np
CLOSED=['PP','lip_press']
OPEN=['aa','E','ih','oh','ou','FF','TH','DD','kk','CH','SS','nn','RR','relaxed_mouth_open','big_smile','teeth_mouth_open']

def calibrate_contact(library,rest,basis):
    poses=[]
    for role in CLOSED+OPEN:
        weights=np.asarray(library['roles'][role]['coefficients'])
        gap=float(np.median(rest+np.einsum('pc,c->p',basis,weights)))
        poses.append(dict(role=role,weights=weights.tolist(),gap=gap,closed=role in CLOSED))
    closed=max(p['gap'] for p in poses if p['closed']);opened=min(p['gap'] for p in poses if not p['closed'])
    if not opened>closed:raise ValueError('Closed/open viseme contact features overlap; calibration needs review')
    capture=dict(full=(closed+opened)/2,release=opened,closedAnchor=closed,openAnchor=opened,
        method='midpoint margin between closed and nearest open viseme; smooth capture to open boundary',
        source='geometry-matched actual-avatar viseme library; PP/lip_press versus open speech poses')
    return capture,poses
