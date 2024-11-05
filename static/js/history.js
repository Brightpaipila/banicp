// history.js

function retrieveHistory() {
    const keyword = document.getElementById('keyword').value;
    const entry = localStorage.getItem(keyword);
    const display = document.getElementById('historyDisplay');

    if (entry) {
        const historyData = JSON.parse(entry);
        display.innerHTML = `
            <p><strong>First Name:</strong> ${historyData.firstName}</p>
            <p><strong>Last Name:</strong> ${historyData.lastName}</p>
            <p><strong>Age:</strong> ${historyData.age}</p>
            <p><strong>Phone:</strong> ${historyData.phone}</p>
        `;
    } else {
        display.innerHTML = "<p>No data found for the provided keyword.</p>";
    }
}

function clearInputs() {
    document.getElementById('keyword').value = '';
    document.getElementById('historyDisplay').innerHTML = '';
}
