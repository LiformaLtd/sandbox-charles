import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

export function createViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x101419);
  container.append(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', 'Interactive Gaussian splat');
  const scene = new THREE.Scene();
  scene.add(new SparkRenderer({ renderer }));
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  let mesh;
  let center = new THREE.Vector3();
  let distance = 3;

  function reset() {
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(distance, 0, 0));
    controls.update();
  }

  function clear() {
    if (!mesh) return;
    scene.remove(mesh);
    mesh.dispose();
    mesh = undefined;
  }

  new ResizeObserver(() => {
    const { clientWidth: width, clientHeight: height } = container;
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }).observe(container);
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
  reset();

  return {
    clear,
    reset,
    async show(blob) {
      clear();
      const next = new SplatMesh({ fileBytes: await blob.arrayBuffer(), fileName: 'splat.ply' });
      try {
        await next.initialized;
        next.rotation.x = -Math.PI / 2;
        next.updateMatrixWorld(true);
        const bounds = next.getBoundingBox(true).applyMatrix4(next.matrixWorld);
        center = bounds.getCenter(new THREE.Vector3());
        const size = bounds.getSize(new THREE.Vector3());
        distance = 0.65 * Math.max(size.y, size.z / camera.aspect) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) + size.x / 2;
        scene.add(next);
        mesh = next;
        reset();
      } catch (error) {
        next.dispose();
        throw error;
      }
    },
  };
}
