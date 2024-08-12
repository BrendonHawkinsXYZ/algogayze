document.addEventListener('DOMContentLoaded', async function () {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let model;

    // Load the YOLO-based model
    async function loadModel() {
        model = await tf.loadGraphModel('model/model.json');
        console.log('Model loaded successfully.');
    }

    // Call loadModel to load the model when the DOM content is fully loaded
    loadModel();

    document.getElementById('upload').addEventListener('change', handleImageUpload);

    async function handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = async function(e) {
                const img = new Image();
                img.onload = async function() {
                    setupCanvas(img);

                    // Ensure the model is loaded before running predictions
                    if (!model) {
                        console.error("Model not loaded yet.");
                        return;
                    }

                    // Resize the image to the model's expected input size
                    const inputSize = 1280;
                    const resizedImageTensor = tf.image.resizeBilinear(tf.browser.fromPixels(canvas), [inputSize, inputSize]).expandDims(0).toFloat();
                    
                    // Run the prediction
                    const predictions = await model.executeAsync(resizedImageTensor);

                    // Randomly choose between refined and chaotic detection
                    const randomChoice = Math.random();
                    let faceCount = 0;

                    if (randomChoice < 0.5) {
                        faceCount = processPredictionsRefined(predictions);
                    } else {
                        faceCount = processPredictionsChaotic(predictions);
                    }

                    // Overlay the number of faces detected on the canvas
                    addTextToCanvas(`DETECTED ${faceCount} FACE(S)`);

                    // Clean up
                    tf.dispose([resizedImageTensor, predictions]);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    }

    function setupCanvas(img) {
        const imgAspectRatio = img.width / img.height;
    
        // Calculate the best fit size within the 70vw by 70vh canvas area
        let newWidth = window.innerWidth * 0.7;
        let newHeight = newWidth / imgAspectRatio;
    
        if (newHeight > window.innerHeight * 0.7) {
            newHeight = window.innerHeight * 0.7;
            newWidth = newHeight * imgAspectRatio;
        }
    
        // Set the canvas drawing surface size to match the image
        canvas.width = img.width;
        canvas.height = img.height;
    
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    
        // Disable image smoothing to prevent blurring
        ctx.imageSmoothingEnabled = false;
    
        // Draw the image at its natural size, then scale down using CSS
        ctx.drawImage(img, 0, 0);
    
        // Scale the canvas element using CSS
        canvas.style.width = `${newWidth}px`;
        canvas.style.height = `${newHeight}px`;
    }

    // Refined detection method
    function processPredictionsRefined(predictions) {
        const [boxes, scores, classes] = predictions;

        const detectionThreshold = 0.99;  // Higher threshold for better accuracy
        let faceCount = 0;

        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;

        for (let i = 0; i < scores.shape[1]; i++) {
            if (scores.dataSync()[i] > detectionThreshold) {
                const [y1, x1, y2, x2] = boxes.dataSync().slice(i * 4, (i + 1) * 4);
                const width = (x2 - x1) * canvasWidth;
                const height = (y2 - y1) * canvasHeight;
                const x = x1 * canvasWidth;
                const y = y1 * canvasHeight;

                // Ensure bounding box is within canvas bounds
                if (x >= 0 && y >= 0 && width > 0 && height > 0 && (x + width) <= canvasWidth && (y + height) <= canvasHeight) {
                    drawBoundingBox(x, y, width, height);
                    pixelateFace(x, y, width, height, 50);
                    faceCount++;
                }
            }
        }

        return faceCount;
    }

    // Chaotic detection method
    function processPredictionsChaotic(predictions) {
        const [boxes, scores, classes] = predictions;

        const detectionThreshold = 0.7;  // Lower threshold for more detections
        let faceCount = 0;

        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;

        for (let i = 0; i < scores.shape[1]; i++) {
            if (scores.dataSync()[i] > detectionThreshold) {
                const [y1, x1, y2, x2] = boxes.dataSync().slice(i * 4, (i + 1) * 4);
                const width = (x2 - x1) * canvasWidth;
                const height = (y2 - y1) * canvasHeight;
                const x = x1 * canvasWidth;
                const y = y1 * canvasHeight;

                // Looser bounds, allowing more flexibility
                if (width > 10 && height > 10) { // Only filtering out tiny boxes
                    drawBoundingBox(x, y, width, height);
                    pixelateFace(x, y, width, height, Math.floor(Math.random() * 50) + 10);  // Random pixel size for chaos
                    faceCount++;
                }
            }
        }

        return faceCount;
    }

    function drawBoundingBox(x, y, width, height) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
    }

    function pixelateFace(x, y, width, height, pixelSize) {
        const maxWidth = canvas.width;
        const maxHeight = canvas.height;

        for (let i = y; i < y + height && i < maxHeight; i += pixelSize) {
            for (let j = x; j < x + width && j < maxWidth; j += pixelSize) {
                const r = Math.floor(Math.random() * 256);
                const g = Math.floor(Math.random() * 256);
                const b = Math.floor(Math.random() * 256);

                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(j, i, Math.min(pixelSize, maxWidth - j), Math.min(pixelSize, maxHeight - i));
            }
        }
    }

    function addTextToCanvas(text) {
        // Get image dimensions
        const width = canvas.width;
        const height = canvas.height;
    
        // Calculate the font size to be 10% of the image width
        const fontSize = width * 0.05;
    
        // Define font style and color
        const font = `bold ${fontSize}px Arial`;
        const fontColor = 'white';
        const margin = 20;  // Increase margin to move text slightly up from the bottom
    
        ctx.font = font;
        ctx.fillStyle = fontColor;
    
        // Measure the text width
        const textSize = ctx.measureText(text);
    
        // Calculate the text position to be centered horizontally
        const textX = (width - textSize.width) / 2;
    
        // Vertically position the text near the bottom, with some margin
        const textY = height - margin;
    
        // Draw the text on the canvas
        ctx.fillText(text, textX, textY);
    }

    document.getElementById('download').addEventListener('click', function() {
        const link = document.createElement('a');
        link.download = 'pixelated-image.png';
        link.href = canvas.toDataURL();
        link.click();
    });
});
