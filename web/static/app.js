const video = document.getElementById("video-feed");
const statusBanner = document.getElementById("status");

const showStatus = (text) => {
  statusBanner.textContent = text;
  statusBanner.hidden = false;
};

const hideStatus = () => {
  statusBanner.hidden = true;
};

const reloadStream = () => {
  const cacheBuster = Date.now();
  video.src = `/video-stream?cb=${cacheBuster}`;
};

video.addEventListener("error", () => {
  showStatus("Reconnecting to camera…");
  setTimeout(reloadStream, 1500);
});

video.addEventListener("load", hideStatus);

document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "r") {
    showStatus("Reloading feed…");
    reloadStream();
  }
});

showStatus("Initializing camera…");
