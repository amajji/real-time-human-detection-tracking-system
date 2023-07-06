// File Upload
//
var fontInBase64 = "",
  fileName = "",
  message = document.querySelector("div"),
  txtForPdf = document.querySelector("textarea"),
  errorStr = '<b style="color:red">Please select a font file!</b>';

function readFile() {
  var file = document.querySelector("input[type=file]").files[0],
    reader = new FileReader();

  if (file && file.name.split(".")[1].toLowerCase() != "ttf") {
    message.innerHTML = errorStr;
    return;
  }

  if (txtForPdf.value.replace(/\s+/g, "").length < 1) {
    message.innerHTML = '<b style="color:red">Please write some Text!</b>';
    return;
  }

  reader.onloadend = function () {
    fontInBase64 = reader.result.split(",")[1];
    fileName = file.name.replace(/\s+/g, "-");

    createPDF(fileName, fontInBase64);
  };

  if (file) reader.readAsDataURL(file);
  else message.innerHTML = errorStr;
}

function createPDF(fileName, fontInBase64) {
  var doc = new jsPDF("p", "mm", "a4");
  (fileNameWithoutExtension = fileName.split(".")[0]),
    (lMargin = 15), // left margin in mm
    (rMargin = 15), // right margin in mm
    (pdfInMM = 210); // width of A4 in mm

  doc.addFileToVFS(fileName, fontInBase64);
  doc.addFont(fileName, fileNameWithoutExtension, "normal");

  doc.setFont(fileNameWithoutExtension);
  doc.setFontSize(14);
  var splitParts = doc.splitTextToSize(
    txtForPdf.value,
    pdfInMM - lMargin - rMargin
  );
  doc.text(15, 15, splitParts);

  doc.save("test.pdf");
}

(function () {
  window.supportDrag = (function () {
    let div = document.createElement("div");
    return (
      ("draggable" in div || ("ondragstart" in div && "ondrop" in div)) &&
      "FormData" in window &&
      "FileReader" in window
    );
  })();

  let input = document.getElementById("js-file-input");

  if (!supportDrag) {
    document.querySelectorAll(".has-drag")[0].classList.remove("has-drag");
  }

  input.addEventListener(
    "change",
    function (e) {
      document.getElementById("js-file-name").innerHTML = this.files[0].name;
      document
        .querySelectorAll(".file-input")[0]
        .classList.remove("file-input--active");
    },
    false
  );

  if (supportDrag) {
    input.addEventListener("dragenter", function (e) {
      document
        .querySelectorAll(".file-input")[0]
        .classList.add("file-input--active");
    });

    input.addEventListener("dragleave", function (e) {
      document
        .querySelectorAll(".file-input")[0]
        .classList.remove("file-input--active");
    });
  }
})();
