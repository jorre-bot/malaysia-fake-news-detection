async function analyzeNews() {
    const newsText = document.getElementById('newsText').value.trim();
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const loadingDiv = document.getElementById('loading');
    const predictionSpan = document.getElementById('prediction');
    const confidenceSpan = document.getElementById('confidence');

    // Reset display
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    
    try {
        // Show loading
        loadingDiv.style.display = 'block';

        // Get CSRF token from meta tag
        const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
        if (!csrfToken) {
            throw new Error('CSRF token not found');
        }

        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
            },
            credentials: 'same-origin',
            body: JSON.stringify({ text: newsText })
        });

        if (!response.ok) {
            throw new Error('Error analyzing news');
        }

        const data = await response.json();

        // Hide loading
        loadingDiv.style.display = 'none';

        if (data.status === 'success') {
            // Update prediction
            predictionSpan.textContent = data.prediction;
            predictionSpan.className = 'prediction-text ' + data.prediction.toLowerCase();
            
            // Update confidence
            const confidence = (data.confidence * 100).toFixed(2);
            confidenceSpan.textContent = `${confidence}%`;
            
            // Show result
            resultDiv.style.display = 'block';
        } else {
            throw new Error(data.error || 'Error analyzing news');
        }
    } catch (error) {
        // Hide loading
        loadingDiv.style.display = 'none';
        
        // Show error
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
    }
}

function clearText() {
    document.getElementById('newsText').value = '';
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display = 'none';
} 