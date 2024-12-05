// Obtener elementos
const dropArea = document.querySelector('.file-upload-container');
const fileInput = document.getElementById('csvUpload');
const fileNameElement = document.getElementById('fileName');
const uploadButton = document.getElementById('uploadButton');

// Validar el archivo
function validateFile(file) {
    const validTypes = ['text/csv', 'application/vnd.ms-excel']; // Tipos MIME para CSV
    if (file && validTypes.includes(file.type)) {
        fileNameElement.textContent = "File loaded: " + file.name;
        fileNameElement.style.color = "#28a745"; // Verde (archivo válido)
        uploadButton.disabled = false; // Habilitar el botón
    } else {
        fileNameElement.textContent = "Error: Only CSV files are allowed.";
        fileNameElement.style.color = "#dc3545"; // Rojo (error)
        fileInput.value = ""; // Limpiar el campo de archivo
        uploadButton.disabled = true; // Deshabilitar el botón
    }
}

// Manejar el evento onchange del input
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    validateFile(file);
});

// Manejar eventos de drag and drop
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('dragging');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragging');
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files; // Asignar archivo al input
    validateFile(file);
});