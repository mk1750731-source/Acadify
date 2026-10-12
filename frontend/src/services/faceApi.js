import * as faceapi from 'face-api.js';

const MODEL_URL = '/models';
let loadPromise = null;

export function loadModels() {
  if (!loadPromise) {
    loadPromise = Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ]);
  }
  return loadPromise;
}

/**
 * Wait until video has actual frame data (readyState >= 2 and dimensions set).
 */
function waitForVideoReady(videoEl, timeout = 5000) {
  return new Promise((resolve, reject) => {
    if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
      resolve();
      return;
    }
    const start = Date.now();
    const check = setInterval(() => {
      if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
        clearInterval(check);
        resolve();
      } else if (Date.now() - start > timeout) {
        clearInterval(check);
        reject(new Error('Video not ready'));
      }
    }, 100);
  });
}

/**
 * Capture current video frame to a canvas.
 * Uses willReadFrequently to suppress the getImageData warning.
 */
function captureCanvas(videoEl) {
  const canvas = document.createElement('canvas');
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  // willReadFrequently fixes the repeated getImageData warning
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/**
 * Compute a 128-dim face descriptor from a video element.
 * Returns float array or null if no face detected.
 */
export async function getEmbedding(videoEl) {
  await loadModels();
  await waitForVideoReady(videoEl);

  const canvas = captureCanvas(videoEl);

  // Try SSD MobileNet first
  let detection = await faceapi
    .detectSingleFace(canvas, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
    .withFaceLandmarks()
    .withFaceDescriptor();

  // Fallback to TinyFaceDetector
  if (!detection) {
    detection = await faceapi
      .detectSingleFace(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 }))
      .withFaceLandmarks()
      .withFaceDescriptor();
  }

  if (!detection) return null;
  return Array.from(detection.descriptor);
}
