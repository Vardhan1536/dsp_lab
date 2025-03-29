const apiBaseUrl = "https://your-server.com";  // Change this after deployment

async function predictStance() {
    const tweet = document.getElementById('tweet').value;
    const target = document.getElementById('target').value;

    if (!tweet || !target) {
        alert("Please enter both tweet and target.");
        return;
    }

    const response = await fetch(`${apiBaseUrl}/predict_stance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tweet, target })
    });

    const data = await response.json();
    if (data.predicted_stance) {
        document.getElementById('prediction').innerText = `Predicted Stance: ${data.predicted_stance}`;
        document.getElementById('prediction').setAttribute('data-stance', data.predicted_stance);
        document.getElementById('feedback-section').style.display = "block";
    }
}

async function submitFeedback(correct) {
    const tweet = document.getElementById('tweet').value;
    const target = document.getElementById('target').value;
    const predictedStance = document.getElementById('prediction').getAttribute('data-stance');
    const correctStance = document.getElementById('correct-stance').value;

    const feedbackData = {
        tweet,
        target,
        predicted_stance: predictedStance,
        feedback: correct ? "Correct" : "Incorrect",
        correct_stance: correct ? predictedStance : correctStance
    };

    await fetch(`${apiBaseUrl}/store_feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(feedbackData)
    });

    alert("Feedback submitted!");
}
