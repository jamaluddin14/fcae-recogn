const express = require('express');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');
const faceapi = require('face-api.js');
const { Canvas, createCanvas, Image, loadImage } = require('canvas');
const tf = require('@tensorflow/tfjs-node');

// Set the TensorFlow CPU logging level using an environment variable
process.env.TF_CPP_MIN_LOG_LEVEL = '3';

const canvas = createCanvas(1, 1);
const faceapiCanvas = canvas.getContext('2d');
faceapi.env.monkeyPatch({ Canvas, Image });

const CONFIG = {
    PROXY_PORT: 5002,
    WS_PORT: 8080,
    FRAME_RATE: 10,
    MAX_CONCURRENT_PROCESSES: Math.max(1, os.cpus().length - 1),
    TEMP_DIR: path.join(os.tmpdir(), 'face-detection'),
    MODEL_PATH: './models'
};

if (!fs.existsSync(CONFIG.TEMP_DIR)) {
    fs.mkdirSync(CONFIG.TEMP_DIR);
}

const app = express();
const wsServer = new WebSocket.Server({ port: CONFIG.WS_PORT });

app.use(cors({
    origin: '*',
}));
const proxyOptions = {
    target: 'http://intravel.amagi.tv',
    changeOrigin: true,
    pathRewrite: {
        '^/hls-proxy/': '/',
    },
    onProxyRes: function (proxyRes, req, res) {
        proxyRes.headers['Access-Control-Allow-Origin'] = '*';
    },
};

app.use('/hls-proxy', createProxyMiddleware(proxyOptions));
const FACE_DETECTION_OPTIONS = {
};

let modelsLoaded = false;

async function loadFaceDetectionModels() {
    if (modelsLoaded) return;

    console.time('Model Loading');
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(CONFIG.MODEL_PATH);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(CONFIG.MODEL_PATH);
    await faceapi.nets.tinyFaceDetector.loadFromDisk(CONFIG.MODEL_PATH);
    modelsLoaded = true;
    console.timeEnd('Model Loading');
}

async function detectFaces(imageBuffer) {
    try {
        const results = await faceapi
            .detectAllFaces(imageBuffer, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks();
        console.log('Detected faces:', results.length);
        console.log('Detected faces:', results);
        const faces = results;

        return {
            faces,
        };
    } catch (error) {
        console.error('Face detection error:', error);
        return { faces: [], frame: null };
    }
}

function processVideoStream(videoUrl, ws) {
    // Updated FFmpeg command with compatible pixel format
    const ffmpegProcess = spawn('ffmpeg', [
        '-i', videoUrl,
        '-vf', `fps=${CONFIG.FRAME_RATE}`,
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-pix_fmt', 'yuvj444p',
        '-q:v', '2',
        '-'
    ]);

    let buffer = Buffer.from([]);
    let frameCounter = 0;
    let isProcessing = false;
    let concurrentProcesses = 0;

    ffmpegProcess.stdout.on('data', async (chunk) => {
        // Limit the number of concurrent processes
        if (concurrentProcesses >= CONFIG.MAX_CONCURRENT_PROCESSES) return;

        if (isProcessing) return; // Skip if still processing previous frame

        buffer = Buffer.concat([buffer, chunk]);

        const startMarker = Buffer.from([0xFF, 0xD8]);
        const endMarker = Buffer.from([0xFF, 0xD9]);

        let startIdx = 0;
        let endIdx = 0;

        while ((startIdx = buffer.indexOf(startMarker, endIdx)) !== -1) {
            endIdx = buffer.indexOf(endMarker, startIdx);
            if (endIdx === -1) break;
            endIdx += 2;

            try {
                isProcessing = true;
                concurrentProcesses++;
                const frameBuffer = buffer.slice(startIdx, endIdx);
                const tensor3d = tf.node.decodeJpeg(frameBuffer);
                const { faces, frame } = await detectFaces(tensor3d);
                console.log(faces, frame);
                if (faces.length > 0 && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'faces_detected',
                        frameNumber: frameCounter,
                        faces,
                        frame
                    }));
                }

                frameCounter++;
                isProcessing = false;
                concurrentProcesses--;
            } catch (error) {
                console.error('Frame processing error:', error);
                isProcessing = false;
                concurrentProcesses--;
            }
        }

        if (endIdx > 0) {
            buffer = buffer.slice(endIdx);
        }
    });

    // Improved error handling for FFmpeg
    ffmpegProcess.stderr.on('data', (data) => {
        const message = data.toString();
        // Log initialization messages at debug level
        if (message.includes('Stream mapping:') ||
            message.includes('Output #0') ||
            message.includes('Press [q] to stop')) {
            console.debug(message.trim());
        }
        // Log actual errors at error level
        else if (!message.includes('frame=') && !message.includes('fps=')) {
            console.error(`FFmpeg error: ${message.trim()}`);
        }
    });

    ffmpegProcess.on('error', (error) => {
        console.error('FFmpeg process error:', error);
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Video processing error occurred'
            }));
        }
    });

    ffmpegProcess.on('close', (code) => {
        console.log(`FFmpeg process exited with code ${code}`);
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'stream_ended' }));
        }
    });

    return ffmpegProcess;
}

// Rest of the server code remains the same...
wsServer.on('connection', async (ws) => {
    await loadFaceDetectionModels();
    let activeProcess = null;

    ws.on('message', async (msg) => {
        try {
            const { videoUrl } = JSON.parse(msg);
            console.log('Processing video URL:', videoUrl);

            if (activeProcess) {
                console.log('Killing existing process');
                activeProcess.kill();
            }

            activeProcess = processVideoStream(videoUrl, ws);
        } catch (error) {
            console.error('WebSocket message error:', error);
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'error',
                    message: 'Failed to process video stream'
                }));
            }
        }
    });

    ws.on('close', () => {
        if (activeProcess) {
            activeProcess.kill();
        }
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        if (activeProcess) {
            activeProcess.kill();
        }
    });
});

// Server startup and error handling remain the same...
function startServer() {
    loadFaceDetectionModels().catch(console.error);

    app.listen(CONFIG.PROXY_PORT, () => {
        console.log(`Proxy server running on port ${CONFIG.PROXY_PORT}`);
    });

    console.log(`WebSocket server running on port ${CONFIG.WS_PORT}`);
}

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection:', reason);
});

process.on('SIGTERM', () => {
    console.log('Graceful shutdown initiated');
    wsServer.close();
    process.exit(0);
});

startServer();