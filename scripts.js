document.getElementById('token-slider').addEventListener('input', function(e) {
    document.getElementById('token-slider-label').textContent = `Number of tokens to generate: ${e.target.value}`;
});

document.getElementById('generate-btn').addEventListener('click', function() {
    const prompt = document.getElementById('prompt-input').value;
    const tokens = document.getElementById('token-slider').value;
    generateText(prompt, tokens);
});

function updateText(newText) {
    const textOutput = document.getElementById('text-output');
    textOutput.textContent += newText;
    textOutput.scrollTop = textOutput.scrollHeight; // Scroll to the bottom to show latest text.
}

async function generateText(prompt, tokens) {
    const onnxSession = new onnx.InferenceSession();
    await onnxSession.loadModel("./models/char_tokens/char_level.onnx");

    // Create input tensor
    const inputTensor = new onnx.Tensor(new Int32Array(prompt.length), "int32");
    // load a dict that maps characters to indices
    const itos = await fetch('./models/char_level/itos.json').then(res => res.json());
    console.log(itos);
    const stoi = await fetch('./models/char_level/stoi.json').then(res => res.json());
    console.log(stoi);

    //The prompt is a str we need to convert it to a tensor with each character mapped to an index with stoi
    for (let i = 0; i < prompt.length; i++) {
        inputTensor.data[i] = stoi[prompt[i]];
    }

    // Run model with input
    const outputMap = await onnxSession.run({ input: inputTensor });
    const outputTensor = outputMap.get('output');
    console.log(outputTensor)
    // Assuming the model outputs one token at a time and takes the last output as the next input
    let generatedText = '';
    for (let i = 0; i < tokens; i++) {
        // Get the model's prediction
        const prediction = /* get the prediction from outputTensor */
        
        // Add the prediction to the generated text
        generatedText += prediction;
        
        // Update the generated text on the page
        updateText(prediction);
        
        // Prepare the next input (you will need to implement the logic to convert the generated text to a new input tensor)
        // const nextInputTensor = ...;
        
        // Run model with the new input
        // outputMap = await onnxSession.run({ input: nextInputTensor });
        // outputTensor = outputMap.get('output');
    }
}

