async function upload() {
    const file = document.getElementById("fileInput").files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (data.image) {
        document.getElementById("resultImage").src = data.image;
    }
}