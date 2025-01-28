// Classes correspondant à FashionMNIST
const classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
];

// Charger le modèle ONNX
let session;
(async () => {
    session = await ort.InferenceSession.create('./fashion_mnist.onnx');
    console.log("Model loaded successfully");
})();

// Fonction pour appliquer la transformation softmax
function softmax(logits) {
    const maxLogit = Math.max(...logits); // Éviter les débordements numériques
    const exps = logits.map(logit => Math.exp(logit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sumExps); // Retourne des probabilités
}

// Fonction pour prétraiter l'image
async function preprocessImage(file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);

    // Charger l'image dans un canvas
    await new Promise((resolve) => {
        img.onload = resolve;
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d');

    // Redimensionner et convertir en niveaux de gris
    ctx.drawImage(img, 0, 0, 28, 28);
    const imageData = ctx.getImageData(0, 0, 28, 28).data;

    // Extraire les valeurs des pixels (r, g, b) -> niveaux de gris
    const grayscaleData = [];
    for (let i = 0; i < imageData.length; i += 4) {
        const r = imageData[i];
        const g = imageData[i + 1];
        const b = imageData[i + 2];
        const grayscale = (r + g + b) / 3;
        const normalized = (grayscale / 255.0 - 0.5) / 0.5; // Normalisation entre -1 et 1
        grayscaleData.push(normalized);
    }

    // Convertir en tenseur
    const inputTensor = new ort.Tensor('float32', new Float32Array(grayscaleData), [1, 1, 28, 28]);
    return inputTensor;
}

// Fonction pour faire une prédiction
async function predict() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    const inputTensor = await preprocessImage(file);

    // Exécuter le modèle
    const results = await session.run({ input: inputTensor });
    const output = results.output.data;

    // Appliquer Softmax pour obtenir des probabilités
    const probabilities = softmax(output);

    // Trouver la classe avec la probabilité maximale
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const predictedClass = classes[maxIndex];

    // Mettre à jour l'interface utilisateur
    document.getElementById('result').innerText = predictedClass;

    const confidenceList = document.getElementById('confidenceList');
    confidenceList.innerHTML = '';
    probabilities.forEach((probability, idx) => {
        const li = document.createElement('li');
        li.innerText = `${classes[idx]}: ${(probability * 100).toFixed(2)}%`;
        confidenceList.appendChild(li);
    });
}
