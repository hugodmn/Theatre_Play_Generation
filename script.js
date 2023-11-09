let lossChart; // Global variable to hold the chart instance
let currentTextInterval = null; // Global variable to hold the current text interval
const modelData = {};

// Function to load the JSON data for a specific model
function loadModelData(model) {
  return fetch(`data_web_app/${model}.json`)
    .then(response => response.json())
    .then(data => {
      modelData[model] = data; // Store it in the modelData object
      return data; // Return it for immediate use
    });
}

// Function to update the text output
function updateOutput(model, step) {
  const modelDataEntry = modelData[model];
  const index = modelDataEntry.steps.indexOf(step);
  if (index !== -1) {
    const sentence = modelDataEntry.text_generated[index];

    // Clear any existing interval
    if (currentTextInterval !== null) {
      clearInterval(currentTextInterval);
    }

    // Render the sentence character by character
    let displayText = '';
    let i = 0;
    currentTextInterval = setInterval(() => {
      if (i < sentence.length) {
        displayText += sentence[i];
        document.getElementById('text-output').textContent = displayText;
        i++;
      } else {
        clearInterval(currentTextInterval);
        currentTextInterval = null; // Reset the interval variable
      }
    }, 100);
  }
}

// Function to update the chart
function updateChart(modelDataEntry, stepIndex) {
  const ctx = document.getElementById('lossChart').getContext('2d');

  const onPointClick = (event, elements) => {
    if (elements.length > 0) {
      const firstPoint = elements[0];
      const label = lossChart.data.labels[firstPoint.index];
      document.getElementById('step-slider').value = label;
      document.getElementById('step-value').textContent = label;
      updateOutput(document.getElementById('model-selector').value, label);
      highlightSelectedPoint(modelData[document.getElementById('model-selector').value], label); // Call to update the point highlight
    }
  };





  // If the chart already exists, update its data
  if (lossChart) {
    lossChart.data.labels = modelDataEntry.steps; // All the steps you have
    lossChart.data.datasets[0].data = modelDataEntry.train_loss;
    lossChart.data.datasets[1].data = modelDataEntry.test_loss;
    lossChart.options.onClick = onPointClick; // Attach the click event handler
    lossChart.update();
  } else {
    // Create a new chart
    lossChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: modelDataEntry.steps, // All the steps you have
        datasets: [
          {
            label: 'Train Loss',
            data: modelDataEntry.train_loss,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
          },
          {
            label: 'Test Loss',
            data: modelDataEntry.test_loss,
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
          }
        ]
      },
      options: {
        onClick: onPointClick, // Attach the click event handler
        scales: {
          x: {
            beginAtZero: true
          },
          y: {
            beginAtZero: true
          }
        },
        elements: {
          point: {
            // We will update this configuration to highlight the selected step
          }
        }
      }
    });
  }

  // Highlight the selected point for the current step
  highlightSelectedPoint(modelDataEntry, modelDataEntry.steps[stepIndex]);
}

// Function to highlight the selected point
function highlightSelectedPoint(modelDataEntry, selectedStep) {
  if (lossChart) {
    // Reset all points to default radius
    lossChart.data.datasets.forEach((dataset) => {
      dataset.pointRadius = new Array(dataset.data.length).fill(3); // Reset radius or initialize if undefined
    });

    // Set the radius of the selected point
    lossChart.data.datasets.forEach((dataset) => {
      const pointIndex = modelDataEntry.steps.findIndex(step => step === selectedStep);
      if (pointIndex !== -1) {
        dataset.pointRadius[pointIndex] = 6; // Highlighted radius
      }
    });

    // Update the chart to reflect the changes
    lossChart.update();
  }
}

// Event listeners for UI interactions
document.getElementById('model-selector').addEventListener('change', function() {
  const selectedModel = this.value;
  const selectedStep = parseInt(document.getElementById('step-slider').value, 10);
  updateOutput(selectedModel, selectedStep);
  updateChart(modelData[selectedModel], modelData[selectedModel].steps.indexOf(selectedStep));
});

document.getElementById('step-slider').addEventListener('input', function() {
  const selectedStep = parseInt(this.value, 10);
  document.getElementById('step-value').textContent = selectedStep;
  const selectedModel = document.getElementById('model-selector').value;
  updateOutput(selectedModel, selectedStep);
  updateChart(modelData[selectedModel], modelData[selectedModel].steps.indexOf(selectedStep));
  highlightSelectedPoint(modelData[selectedModel], selectedStep); // Call to update the point highlight
});

// Load all model data when the app starts
Promise.all([
  loadModelData('char_level_tokenizer'),
  loadModelData('word_level_tokenizer'),
  // Uncomment these if the corresponding JSON files are present and you want to load them
  // loadModelData('word_level_tokenizer'),
  loadModelData('bert_tokenizer')
]).then(() => {
  // Set the initial model and step
  const initialModel = 'char_level_tokenizer';
  const initialStep = 200; // Set the default step to 200
  if (modelData[initialModel] && modelData[initialModel].steps.includes(initialStep)) {
    // Set the slider max and value
    document.getElementById('step-slider').max = 6000;
    document.getElementById('step-slider').value = initialStep;
    document.getElementById('step-value').textContent = initialStep;
    // Update the output and chart for the initial step
    updateOutput(initialModel, initialStep);
    updateChart(modelData[initialModel], modelData[initialModel].steps.indexOf(initialStep));
  } else {
    console.error('Initial step data is not available');
  }
}).catch(error => {
  console.error('Error loading model data:', error);
});