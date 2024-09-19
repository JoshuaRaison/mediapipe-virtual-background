import {
  ImageSegmenter,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const demosSection = document.getElementById("demos");
const backgroundImage = new Image();
backgroundImage.src = "./bg.jpg";

let enableWebcamButton;
let webcamRunning = false;
let lastWebcamTime = -1;
let imageSegmenter;
let labels;
const runningMode = "VIDEO";

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam() {
  if (!imageSegmenter) return;

  webcamRunning = !webcamRunning;
  enableWebcamButton.innerText = webcamRunning
    ? "DISABLE SEGMENTATION"
    : "ENABLE SEGMENTATION";

  if (webcamRunning) {
    const constraints = { video: true };
    video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
    video.addEventListener("loadeddata", predictWebcam);
  }
}

async function createImageSegmenter() {
  const audio = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  imageSegmenter = await ImageSegmenter.createFromOptions(audio, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
      delegate: "GPU",
    },
    runningMode: runningMode,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
  labels = imageSegmenter.getLabels();
  demosSection.classList.remove("invisible");
}

function callbackForVideo(result) {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    backgroundImage,
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );

  const mask = result.categoryMask.getAsFloat32Array();
  const videoFrameData = getVideoFrameData();
  const backgroundFrameData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const backgroundPixels = backgroundFrameData.data;

  for (let i = 0; i < mask.length; i++) {
    const j = i * 4;
    if (mask[i] < 0.5) {
      backgroundPixels[j] = videoFrameData[j];
      backgroundPixels[j + 1] = videoFrameData[j + 1];
      backgroundPixels[j + 2] = videoFrameData[j + 2];
      backgroundPixels[j + 3] = 255;
    } else {
      backgroundPixels[j + 3] = 255;
    }
  }

  canvasCtx.putImageData(backgroundFrameData, 0, 0);

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

function getVideoFrameData() {
  const videoCanvas = document.createElement("canvas");
  const videoCtx = videoCanvas.getContext("2d");
  videoCanvas.width = video.videoWidth;
  videoCanvas.height = video.videoHeight;
  videoCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
  return videoCtx.getImageData(0, 0, video.videoWidth, video.videoHeight).data;
}

async function predictWebcam() {
  if (video.currentTime === lastWebcamTime) {
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
    return;
  }
  lastWebcamTime = video.currentTime;

  if (!imageSegmenter) return;

  await imageSegmenter.setOptions({ runningMode: runningMode });
  const startTimeMs = performance.now();
  imageSegmenter.segmentForVideo(video, startTimeMs, callbackForVideo);
}

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

createImageSegmenter();
