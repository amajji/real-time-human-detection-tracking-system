let ws = new WebSocket("ws://localhost:8000/ws");
let image = document.getElementById("frame");
image.onload = function(){
    URL.revokeObjectURL(this.src); // release the blob URL once the image is loaded
}
ws.onmessage = function(event) {
    image.src = URL.createObjectURL(event.data);
};