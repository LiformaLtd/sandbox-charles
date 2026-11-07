import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadBlendshapeRig } from './blendshape-rig.js';

const teeth = new THREE.MeshStandardMaterial({ color: 0xd9c9b3, roughness: 0.43 });
const materials = {
  skin: new THREE.MeshStandardMaterial({ color: 0xb9846a, roughness: 0.65 }),
  tongue: new THREE.MeshStandardMaterial({ color: 0x863c46, roughness: 0.61 }),
  upperTeeth: teeth,
  lowerTeeth: teeth,
};
const degree = Math.PI / 180;

function smoothStep(value) {
  const t = THREE.MathUtils.clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

async function asset(path, format) {
  const response = await fetch(path);
  if (!response.ok) throw new Error('The local James rig is missing. See the setup instructions.');
  return response[format]();
}

function basis(points) {
  const horizontal = new THREE.Vector3().subVectors(points[1], points[0]).normalize();
  const rear = new THREE.Vector3().lerpVectors(points[0], points[1], 0.5);
  const front = new THREE.Vector3().subVectors(points[4], rear).normalize();
  const up = new THREE.Vector3().crossVectors(front, horizontal).normalize();
  front.crossVectors(horizontal, up);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(horizontal, up, front));
}

function center(points) {
  return points.reduce((sum, point) => sum.add(point), new THREE.Vector3()).divideScalar(points.length);
}

function makeEyeGeometry(geometry, positions, pivot) {
  const colors = new Float32Array(positions.length);
  const sclera = new THREE.Color(0xded9cf);
  const iris = new THREE.Color(0x69644e);
  const pupil = new THREE.Color(0x171516);
  for (let i = 0; i < positions.length; i += 3) {
    const radial = Math.hypot(positions[i] - pivot.x, positions[i + 1] - pivot.y);
    const color = positions[i + 2] < pivot.z || radial > 0.58 ? sclera : radial > 0.24 ? iris : pupil;
    color.toArray(colors, i);
    positions[i] -= pivot.x;
    positions[i + 1] -= pivot.y;
    positions[i + 2] -= pivot.z;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

export async function createJames(viewport, { compact = false } = {}) {
  const [rig, binary] = await Promise.all([asset('/assets/rig.json', 'json'), asset('/assets/rig.bin', 'arrayBuffer')]);
  const skinLength = rig.meshes.skin.vertexCount * 3;
  if (!Number.isInteger(rig.eyeCloseOffset) || rig.eyeCloseOffset + skinLength * 4 > binary.byteLength) {
    throw new Error('The James rig is missing its eye-close pose. Regenerate it with scripts/prepare_rig.py.');
  }
  const eyeClose = new Float32Array(binary, rig.eyeCloseOffset, skinLength);
  const blend = compact && await loadBlendshapeRig([
    skinLength, rig.meshes.tongue.vertexCount * 3, rig.jaw.length * 3,
  ]);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 300);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.domElement.setAttribute('aria-label', 'Animated 3D James face');
  viewport.append(renderer.domElement);

  const head = new THREE.Group();
  head.position.y = -169;
  scene.add(head);
  const faceParts = new THREE.Group();
  head.add(faceParts);
  const meshes = {};
  const rest = {};
  const eyes = [];
  const eyeMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.32 });
  for (const [name, data] of Object.entries(rig.meshes)) {
    const positions = new Float32Array(binary, data.positionOffset, data.vertexCount * 3).slice();
    const indices = new Uint32Array(binary, data.indexOffset, data.indexCount);
    const geometry = new THREE.BufferGeometry();
    const attribute = new THREE.BufferAttribute(positions, 3);
    geometry.setAttribute('position', attribute);
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    const eye = name.endsWith('Eye');
    const mesh = new THREE.Mesh(geometry, materials[name] ?? eyeMaterial);
    if (eye) {
      const pivot = new THREE.Vector3(...data.pivot);
      makeEyeGeometry(geometry, positions, pivot);
      mesh.position.copy(pivot);
      eyes.push({ mesh, rotation: data.rotation, offset: name === 'rightEye' ? 88827 : 88829 });
    }
    if (name === 'skin' || name === 'tongue') {
      rest[name] = positions.slice();
      attribute.setUsage(THREE.DynamicDrawUsage);
    }
    geometry.computeVertexNormals();
    (name === 'skin' ? head : faceParts).add(mesh);
    meshes[name] = mesh;
  }

  scene.add(new THREE.HemisphereLight(0xf5eeff, 0x6d4b56, 2.05));
  function light(color, power, xyz) {
    const lamp = new THREE.DirectionalLight(color, power);
    lamp.position.set(...xyz);
    scene.add(lamp);
  }
  light(0xffe2ca, 2.45, [18, 26, 35]);
  light(0xa1bcff, 1.15, [-22, 9, 12]);
  light(0xf8c0a8, 1.9, [-8, 18, -20]);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 17;
  controls.maxDistance = 135;
  controls.maxPolarAngle = Math.PI * 0.78;
  controls.minPolarAngle = Math.PI * 0.22;

  const neutralJaw = rig.jaw.map((value) => new THREE.Vector3(...value));
  const neutralCenter = center(neutralJaw);
  const neutralInverse = basis(neutralJaw).invert();
  const jaw = neutralJaw.map((point) => point.clone());
  const lower = meshes.lowerTeeth;

  // Approximate the upper-neck pivot in the coordinates of the James mesh.
  const neckPivot = new THREE.Vector3(0, 166, -5.5);
  const neckWeights = new Float32Array(rest.skin.length / 3);
  const faceWeights = new Float32Array(neckWeights.length);
  for (let vertex = 0; vertex < neckWeights.length; vertex++) {
    const i = vertex * 3;
    const x = rest.skin[i];
    const y = rest.skin[i + 1];
    const z = rest.skin[i + 2];
    // The outer shoulders and upper back reach the same height as the neck.
    const outerTorso = Math.max(smoothStep((Math.abs(x) - 4) / 4), smoothStep((-z - 8) / 5));
    const neck = smoothStep((y - 157 - 4 * outerTorso) / 5);
    const shoulderFollow = 0.07 * smoothStep((y - 150) / 7);
    faceWeights[vertex] = neck;
    neckWeights[vertex] = THREE.MathUtils.lerp(shoulderFollow, 1, neck);
  }
  const headRotation = new THREE.Euler(0, 0, 0, 'YXZ');
  const inverseHeadRotation = new THREE.Quaternion();
  const originalPoint = new THREE.Vector3();
  const turnedPoint = new THREE.Vector3();

  function setHeadPose([pitch, yaw, roll] = [0, 0, 0]) {
    headRotation.set(pitch * degree, yaw * degree, roll * degree);
    faceParts.quaternion.setFromEuler(headRotation);
    // The rigid parts share the head pose. Keep their existing local coordinates.
    faceParts.position.copy(neckPivot).applyQuaternion(faceParts.quaternion).negate().add(neckPivot);
  }

  function drawPart(name, frame, offset, blink = 0, turning = false) {
    const geometry = meshes[name].geometry;
    const attribute = geometry.attributes.position;
    const initial = rest[name];
    const skin = name === 'skin';
    for (let i = 0; i < initial.length; i += 3) {
      const weight = skin ? faceWeights[i / 3] : 1;
      for (let j = i; j < i + 3; j++) {
        const speechDelta = (frame?.[offset + j] ?? 0) * weight;
        attribute.array[j] = initial[j] + speechDelta + (blink ? blink * eyeClose[j] : 0);
      }
    }
    if (skin && turning) {
      // Blend the same rigid transform into the skin across the neck.
      for (let vertex = 0; vertex < neckWeights.length; vertex++) {
        const weight = neckWeights[vertex];
        if (!weight) continue;
        const i = vertex * 3;
        originalPoint.fromArray(attribute.array, i);
        turnedPoint.copy(originalPoint).applyQuaternion(faceParts.quaternion).add(faceParts.position);
        originalPoint.lerp(turnedPoint, weight).toArray(attribute.array, i);
      }
    }
    attribute.needsUpdate = true;
    geometry.computeVertexNormals();
  }

  let previousFrame = null;
  let previousBlink = 0;
  let previousRotation;
  let blendDeltas;

  function pose(frame, blink = 0, rotation) {
    const faceChanged = frame !== previousFrame;
    const headChanged = rotation
      ? rotation.some((angle, axis) => angle !== previousRotation?.[axis])
      : Boolean(previousRotation);
    if (!faceChanged && !headChanged && blink === previousBlink) return;

    setHeadPose(rotation);
    if (blend && (faceChanged || !blendDeltas)) blendDeltas = blend(frame);
    const blinkStrength = blend ? blink * (1 - Math.max(frame?.[0] ?? 0, frame?.[7] ?? 0)) : blink;
    drawPart('skin', blend ? blendDeltas[0] : frame, 0, blinkStrength, Boolean(rotation));
    previousBlink = blink;
    previousRotation = rotation;
    if (faceChanged) {
      previousFrame = frame;
      updateJaw(frame);
    }

    inverseHeadRotation.copy(faceParts.quaternion).invert();
    for (const { mesh, rotation, offset } of eyes) {
      mesh.rotation.set(
        (rotation[0] + (blend ? 0 : frame?.[offset] ?? 0)) * degree,
        (rotation[1] + (blend ? 0 : frame?.[offset + 1] ?? 0)) * degree,
        rotation[2] * degree,
      );
      // The eye pivots travel with the head, but gaze stays in scene space.
      mesh.quaternion.premultiply(inverseHeadRotation);
    }
  }

  function updateJaw(frame) {
    drawPart('tongue', blend ? blendDeltas[1] : frame, blend ? 0 : 72006);
    const motion = blend ? blendDeltas[2] : frame;
    const offset = blend ? 0 : 88812;
    for (let i = 0; i < jaw.length; i++) {
      const k = offset + i * 3;
      jaw[i].copy(neutralJaw[i]);
      if (motion) {
        jaw[i].x += motion[k];
        jaw[i].y += motion[k + 1];
        jaw[i].z += motion[k + 2];
      }
    }
    lower.quaternion.copy(basis(jaw).multiply(neutralInverse));
    lower.position.copy(center(jaw)).sub(neutralCenter.clone().applyQuaternion(lower.quaternion));
  }

  function view(mouth = false) {
    controls.target.set(0, mouth ? -1.6 : 0, mouth ? 6 : -2);
    const distance = mouth ? 26 : camera.aspect < 0.95 ? 80 : 69;
    camera.position.set(0, mouth ? -1.6 : 1.4, distance);
    controls.update();
  }

  const resize = new ResizeObserver(() => {
    const { clientWidth: width, clientHeight: height } = viewport;
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  });
  resize.observe(viewport);
  camera.aspect = viewport.clientWidth / viewport.clientHeight;
  camera.updateProjectionMatrix();
  view();
  pose();
  return {
    view,
    pose,
    dispose() {
      resize.disconnect();
      controls.dispose();
      scene.traverse((object) => object.geometry?.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    },
    draw() {
      controls.update();
      renderer.render(scene, camera);
    },
  };
}
