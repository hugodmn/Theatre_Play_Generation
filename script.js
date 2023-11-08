

// Call this function where you're updating the output text and after you have model data
// For example, you could call it at the end of your loadData() function, or after you fetch the data
//let lossChart; // Global variable to hold the chart instance

// Function to update the chart, or create it if it doesn't exist yet
function updateChart(modelData, stepIndex) {
  const ctx = document.getElementById('lossChart').getContext('2d');

  // If the chart already exists, update its data
  if (lossChart) {
    lossChart.data.labels.push(stepIndex * 200); // Assuming steps are multiple of 200
    lossChart.data.datasets[0].data.push(modelData.train_losses[stepIndex]);
    lossChart.data.datasets[1].data.push(modelData.test_losses[stepIndex]);
    lossChart.update();
  } else {
    // Create a new chart
    lossChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: modelData.steps, // All the steps you have
        datasets: [
          {
            label: 'Train Loss',
            data: modelData.train_losses,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
          },
          {
            label: 'Test Loss',
            data: modelData.test_losses,
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
          }
        ]
      },
      options: {
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
  highlightSelectedPoint(stepIndex * 200);
}

// Call this function where you're updating the output text and after you have model data
// For example, you could call it at the end of your loadData() function, or after you fetch the data

// Function to highlight the selected point
function highlightSelectedPoint(selectedStep) {
  if (lossChart) {
    // Reset all points to default radius
    lossChart.data.datasets.forEach((dataset) => {
      dataset.pointRadius = dataset.pointRadius.map(() => 3); // Reset radius
    });

    // Set the radius of the selected point
    lossChart.data.datasets.forEach((dataset) => {
      const pointIndex = dataset.data.findIndex((value, index) => modelData.steps[index] === selectedStep);
      if (pointIndex !== -1) {
        dataset.pointRadius[pointIndex] = 6; // Highlighted radius
      }
    });

    // Update the chart to reflect the changes
    lossChart.update();
  }
}



// Call updateChart somewhere in your code after you have the model data, for example:
// updateChart(mockData['char_level_tokenizer'], 0); // Initial chart render



const mockData = {
    char_level_tokenizer: {
      steps: [200, 400, 600],
      sentences: ["Thou art...", "Wherefore...", "To be..."],
      train_losses: [4.0, 3.5, 3.0],
      test_losses: [4.2, 3.6, 3.1]
    },
    // ... same structure for word_level_tokenizer and bert_tokenizer
  };
  
  // Function to load the JSON data (this would be replaced with an actual AJAX call or fetch)
  function loadData() {
    return new Promise((resolve) => {
      setTimeout(() => resolve(mockData), 1000); // Mock async data loading
    });
  }
  
  // Function to update the text output and graph
  function updateOutput(model, step) {
    const modelData = mockData[model];
    const index = modelData.steps.indexOf(step);
    if (index !== -1) {
      const sentence = modelData.sentences[index];
      // Render the sentence character by character
      let displayText = '';
      let i = 0;
      const interval = setInterval(() => {
        if (i < sentence.length) {
          displayText += sentence[i];
          document.getElementById('text-output').textContent = displayText;
          i++;
        } else {
          clearInterval(interval);
        }
      }, 100);
  
      // Update the loss graph here with the modelData.train_losses[index] and modelData.test_losses[index]
      // This would involve a library like Chart.js or D3.js to render the graph
      // For simplicity, let's just log it to the console
      console.log(`Train Loss: ${modelData.train_losses[index]}, Test Loss: ${modelData.test_losses[index]}`);
    }
  }
  
  // Event listeners for UI interactions
  document.getElementById('model-selector').addEventListener('change', function() {
    const selectedModel = this.value;
    const selectedStep = parseInt(document.getElementById('step-slider').value, 10);
    updateOutput(selectedModel, selectedStep);
    updateChart(mockData[selectedModel], selectedStep);
  });
  
  document.getElementById('step-slider').addEventListener('input', function() {
    const selectedStep = parseInt(this.value, 10);
    document.getElementById('step-value').textContent = selectedStep;
    const selectedModel = document.getElementById('model-selector').value;
    updateOutput(selectedModel, selectedStep);
  });
  
  // Load the data and initialize the app
  loadData().then((data) => {
    // This is where you would update the application with the loaded data
    // For now, we just log it to the console
    console.log(data);
    // Initialize the output with the first model and step
    updateOutput('char_level_tokenizer', 200);
  });


  let lossChart; // Global variable to hold the chart instance

  // Function to update the chart, or create it if it doesn't exist yet
  function updateChart(modelData, stepIndex) {
    const ctx = document.getElementById('lossChart').getContext('2d');
  
    // If the chart already exists, update its data
    if (lossChart) {
      lossChart.data.labels.push(stepIndex * 200); // Assuming steps are multiple of 200
      lossChart.data.datasets[0].data.push(modelData.train_losses[stepIndex]);
      lossChart.data.datasets[1].data.push(modelData.test_losses[stepIndex]);
      lossChart.update();
    } else {
      // Create a new chart
      lossChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: modelData.steps, // All the steps you have
          datasets: [
            {
              label: 'Train Loss',
              data: modelData.train_losses,
              borderColor: 'rgb(255, 99, 132)',
              backgroundColor: 'rgba(255, 99, 132, 0.5)',
            },
            {
              label: 'Test Loss',
              data: modelData.test_losses,
              borderColor: 'rgb(54, 162, 235)',
              backgroundColor: 'rgba(54, 162, 235, 0.5)',
            }
          ]
        },
        options: {
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
    highlightSelectedPoint(stepIndex * 200);
  }
  
  // Call this function where you're updating the output text and after you have model data
  // For example, you could call it at the end of your loadData() function, or after you fetch the data
  
  // Function to highlight the selected point
  function highlightSelectedPoint(selectedStep) {
    if (lossChart) {
      // Reset all points to default radius
      lossChart.data.datasets.forEach((dataset) => {
        dataset.pointRadius = dataset.pointRadius.map(() => 3); // Reset radius
      });
  
      // Set the radius of the selected point
      lossChart.data.datasets.forEach((dataset) => {
        const pointIndex = dataset.data.findIndex((value, index) => modelData.steps[index] === selectedStep);
        if (pointIndex !== -1) {
          dataset.pointRadius[pointIndex] = 6; // Highlighted radius
        }
      });
  
      // Update the chart to reflect the changes
      lossChart.update();
    }
  }
  
  // Call updateChart somewhere in your code after you have the model data, for example:
  // updateChart(mockData['char_level_tokenizer'], 0); // Initial chart render
  


