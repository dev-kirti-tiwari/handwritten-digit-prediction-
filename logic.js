// --- 1. Canvas Setup & Drawing Logic ---
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictionText = document.getElementById('prediction');

let isDrawing = false;

// Initialize canvas with a black background (crucial for MNIST)
function initCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
initCanvas();

// Drawing settings
ctx.strokeStyle = 'white'; // White pen on black background
ctx.lineWidth = 18;        // Thick line to mimic MNIST strokes
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Mouse/Touch Events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing) return;
    
    // Calculate standard mouse position
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

clearBtn.addEventListener('click', () => {
    initCanvas();
    predictionText.innerText = '-';
});

// --- 2. Preprocessing & Prediction Logic ---

async function predictDigit() {
    predictionText.innerText = 'Thinking...';

    // A. PREPROCESSING
    // 1. Shrink the 280x280 canvas to 28x28 using a hidden canvas
    const hiddenCanvas = document.getElementById('hiddenCanvas');
    const hiddenCtx = hiddenCanvas.getContext('2d');
    hiddenCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // 2. Extract the raw pixel data from the 28x28 canvas
    const imgData = hiddenCtx.getImageData(0, 0, 28, 28).data;
    
    // 3. Convert to grayscale and Normalize to match PyTorch
    // Our PyTorch transform was: Normalize((0.5,), (0.5,))
    // Math: (pixel_value / 255.0 - 0.5) / 0.5
    const inputFloatArray = new Float32Array(28 * 28);
    
    for (let i = 0; i < 28 * 28; i++) {
        // imgData contains RGBA, so we jump by 4 to get just the Red channel
        // Since it's black and white, Red = Green = Blue.
        const pixelValue = imgData[i * 4]; 
        
        // Normalize
        const normalized = (pixelValue / 255.0 - 0.5) / 0.5;
        inputFloatArray[i] = normalized;
    }

    // B. RUNNING THE MODEL
    try {
        // 1. Create ONNX Session (loads the file)
        const session = await ort.InferenceSession.create('./mnist_model.onnx');
        
        // 2. Prepare the tensor (Batch:1, Channel:1, Height:28, Width:28)
        const tensor = new ort.Tensor('float32', inputFloatArray, [1, 1, 28, 28]);
        
        // 3. Feed it to the model (Note: 'input' must match the input_names in Python)
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        
        // 4. Get output array (10 numbers representing probabilities for 0-9)
        const outputArray = results.output.data;
        
        // 5. Find the index with the highest probability (argmax)
        let maxIndex = 0;
        let maxValue = outputArray[0];
        
        for (let i = 1; i < outputArray.length; i++) {
            if (outputArray[i] > maxValue) {
                maxValue = outputArray[i];
                maxIndex = i;
            }
        }
        
        // Display result
        predictionText.innerText = maxIndex;

    } catch (error) {
        console.error(error);
        predictionText.innerText = "Error!";
    }
}

predictBtn.addEventListener('click', predictDigit);