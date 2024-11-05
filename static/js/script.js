// script.js

// Function to save user credentials
function saveCredentials() {
    const firstName = document.getElementById('firstName').value;
    const lastName = document.getElementById('lastName').value;
    const age = document.getElementById('age').value;
    const phone = document.getElementById('phone').value;
    const keyword = document.getElementById('keyword').value;

    // Check for empty fields
    if (!firstName || !lastName || !age || !phone || !keyword) {
        alert("Please fill all fields.");
        return;
    }

    // Validate phone number format (example)
    const phoneRegex = /^\d{10}$/; // Adjust regex as needed
    if (!phoneRegex.test(phone)) {
        alert("Please enter a valid 10-digit phone number.");
        return;
    }

    // Create an entry and store it in localStorage with the keyword as the key
    const historyEntry = { firstName, lastName, age, phone };
    localStorage.setItem(keyword, JSON.stringify(historyEntry)); // Saving entry with keyword

    alert("Credentials saved successfully.");
    clearInputs(); // Clear inputs after saving
}

// Function to clear input fields
function clearInputs() {
    document.getElementById('firstName').value = '';
    document.getElementById('lastName').value = '';
    document.getElementById('age').value = '';
    document.getElementById('phone').value = '';
    document.getElementById('keyword').value = '';
}

// Function to open a modal (if applicable)
// Note: This code would require a modal structure to work correctly.
// function openModal() { ... }
// function closeModal() { ... }

// Function to handle image prediction
async function sendImageToServer(imageData) {
    try {
        const response = await fetch('http://your_fastapi_server_url/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
        });
        const data = await response.json();
        console.log(data);
        // Handle the response from the server as needed
    } catch (error) {
        console.error('Error:', error);
    }
}
