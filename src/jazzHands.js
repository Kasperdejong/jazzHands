import {Hands} from '@mediapipe/hands';
import {Camera} from '@mediapipe/camera_utils';
import {HAND_CONNECTIONS} from '@mediapipe/hands';
import {drawConnectors, drawLandmarks} from '@mediapipe/drawing_utils';

import * as tf from '@tensorflow/tfjs';

// vars
const video = document.getElementById('input_video');
const canvas = document.getElementById('hand_canvas');
const ctx = canvas.getContext('2d');
const startWebcamButton = document.getElementById('start-webcam')
const beginSound = new Audio('/sounds/charge.mp3');


const resultDisplay = document.getElementById('gesture-target');

let handLandmarks = [];
let gameRunning = false;
let currentGesture = null;



const gestures = [
    { label: 'openHand', emoji: '✋', sound: 'sounds/kickdrum.mp3' },
    { label: 'fist', emoji: '👊', sound: 'sounds/bonk.mp3' },
    { label: 'thumbsUp', emoji: '👍', sound: 'sounds/bassNote.mp3' },
    { label: 'DevilsHorns', emoji: '🤟', sound: 'sounds/guitarRiff.mp3' },
    { label: 'Peace', emoji: '✌️', sound: 'sounds/partyTrumpet.mp3' }
];
const failSound = new Audio('sounds/clarinetFail.mp3');

const neuralNetwork = ml5.neuralNetwork({
    task: 'classification',
    debug: true
});

// Load the trained model
async function loadModel() {
    await tf.ready(); // Ensure TensorFlow.js is ready before loading the model
    neuralNetwork.load('./model.json', () => {
        console.log('Model loaded!');
    });
}

function normalizeHandData() {
    const flatHandArray = [];

    if (handLandmarks[0].length > 0) {
        const wrist = handLandmarks[0][0]; // Wrist is always index 0

        for (let i = 0; i < handLandmarks[0].length; i++) {
            const landmark = handLandmarks[0][i];

            const x = landmark.x - wrist.x;
            const y = landmark.y - wrist.y;
            const z = landmark.z - wrist.z;

            flatHandArray.push(x, y, z);
        }
        return flatHandArray;
    }
    return null;
}

function startGameLoop() {
    gameRunning = true;
    nextRound();
}

function nextRound() {
    currentGesture = gestures[Math.floor(Math.random() * gestures.length)];
    resultDisplay.innerText = currentGesture.emoji;

    setTimeout(() => {
        classifyCurrentHand();
    }, 1500);
}


async function classifyCurrentHand() {
    if (handLandmarks.length === 0 || handLandmarks[0].length === 0) {
        console.warn('No hands detected');
        failSound.play(); // Play the fail sound if no hands are detected

        // Proceed to the next round even if no hand is detected
        if (gameRunning) {
            setTimeout(nextRound, 1500);
        }
        return;
    }
    // Classify the current hand data using the neural network
    const results = await neuralNetwork.classify(normalizeHandData());

    if (!results || results.length === 0) {
        console.warn('No hand data to classify');
        failSound.play(); // Play fail sound if no results
        return;
    }

    // Log the raw results for inspection
    console.log('Results:', results);

    // Get the predicted label from the first result and normalize both labels
    const predictedLabel = results[0].label.trim().toLowerCase(); // Normalize by trimming and converting to lowercase
    const expectedLabel = currentGesture.label.trim().toLowerCase();

    // Log to check what the model is outputting and the expected gesture label
    console.log(`Predicted: ${predictedLabel}, Expected: ${expectedLabel}`);

    // Check if the predicted label matches the current gesture label
    if (predictedLabel === expectedLabel) {
        // Play the corresponding sound for the gesture
        new Audio(currentGesture.sound).play();
    } else {
        // Play the fail sound if the label doesn't match
        failSound.play();
    }

    // If the game is running, move to the next round
    if (gameRunning) {
        setTimeout(nextRound, 1500);
    }
}

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
});

hands.onResults((results) => {
    handLandmarks = results.multiHandLandmarks || [];
    drawLandmarksFunc();
});

// Draw loop
function drawLandmarksFunc() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const hand of handLandmarks) {
        drawConnectors(ctx, hand, HAND_CONNECTIONS, {
            color: 'blue',
            lineWidth: 2
        });

        drawLandmarks(ctx, hand, {
            color: 'green',
            lineWidth: 1
        });
    }
}

// === Camera Setup ===
const camera = new Camera(video, {
    onFrame: async () => {
        await hands.send({image: video});
    },
    width: 640,
    height: 480,
});

async function init() {
    await camera.start();
}

// === Start Webcam and Game ===
startWebcamButton.addEventListener('click', () => {
    loadModel();
    init();
    beginSound.play();
    startGameLoop();
});


