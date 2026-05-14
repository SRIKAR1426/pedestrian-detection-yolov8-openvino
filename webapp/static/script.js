// ── Tab switching ────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  event.target.classList.add('active');
  if (tab === 'dataset') loadDatasetImages();
}

// ── Image Detection ──────────────────────────────
let imgFile = null;

function handleImgUpload(event) {
  imgFile = event.target.files[0];
  if (!imgFile) return;
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('img-preview');
    preview.src = e.target.result;
    preview.classList.remove('hidden');
    document.getElementById('img-drop-zone').style.display = 'none';
    document.getElementById('btn-detect-img').disabled = false;
  };
  reader.readAsDataURL(imgFile);
}

function handleImgDrop(event) {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) {
    document.getElementById('img-input').files = event.dataTransfer.files;
    handleImgUpload({ target: { files: [file] } });
  }
}

function detectImage() {
  if (!imgFile) return;

  const btn = document.getElementById('btn-detect-img');
  btn.disabled = true;
  btn.textContent = 'Detecting...';

  document.getElementById('img-result-zone').innerHTML =
    '<div class="spinner"></div>';
  document.getElementById('stat-count').textContent = '—';
  document.getElementById('stat-time').textContent  = '—';
  document.getElementById('stat-conf').textContent  = '—';

  const formData = new FormData();
  formData.append('file', imgFile);

  fetch('/detect_image', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }
      document.getElementById('img-result-zone').innerHTML =
        `<img src="data:image/jpeg;base64,${data.detected}" style="width:100%;border-radius:8px;">`;
      document.getElementById('stat-count').textContent = data.count;
      document.getElementById('stat-time').textContent  = data.inference_ms + 'ms';
      document.getElementById('stat-conf').textContent  = data.avg_conf + '%';
    })
    .catch(err => alert('Error: ' + err))
    .finally(() => {
      btn.disabled = false;
      btn.textContent = 'Detect Pedestrians';
    });
}

// ── Video Detection ──────────────────────────────
let vidFile = null;

function handleVidUpload(event) {
  vidFile = event.target.files[0];
  if (!vidFile) return;
  const preview = document.getElementById('vid-preview');
  preview.src = URL.createObjectURL(vidFile);
  preview.classList.remove('hidden');
  document.getElementById('vid-drop-zone').style.display = 'none';
  document.getElementById('btn-detect-vid').disabled = false;
}

function handleVidDrop(event) {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) {
    handleVidUpload({ target: { files: [file] } });
  }
}

function detectVideo() {
  if (!vidFile) return;

  const btn = document.getElementById('btn-detect-vid');
  btn.disabled = true;
  btn.textContent = 'Processing...';

  const progress   = document.getElementById('vid-progress');
  const fill       = document.getElementById('progress-fill');
  const progressTxt= document.getElementById('progress-text');
  progress.classList.remove('hidden');

  document.getElementById('vid-result-zone').innerHTML =
    '<div class="spinner"></div>';

  // Animate progress bar
  let pct = 0;
  const interval = setInterval(() => {
    pct = Math.min(pct + Math.random() * 3, 90);
    fill.style.width = pct + '%';
    progressTxt.textContent = `Processing video... ${Math.round(pct)}%`;
  }, 500);

  const formData = new FormData();
  formData.append('file', vidFile);

  fetch('/detect_video', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      clearInterval(interval);
      fill.style.width = '100%';
      progressTxt.textContent = 'Done!';

      if (data.error) {
        alert(data.error);
        return;
      }

      document.getElementById('vid-result-zone').innerHTML = `
  <div style="padding:16px;text-align:center">
    <p style="color:#00d4aa;font-size:14px;margin-bottom:12px">
      Video processed successfully!
    </p>
    <a href="/download_video/${data.output_video}"
       class="btn-download">
      Download Detected Video
    </a>
  </div>`;
      document.getElementById('vid-frames').textContent = data.total_frames;
      document.getElementById('vid-dets').textContent   = data.total_dets;
      document.getElementById('vid-fps').textContent    = data.avg_fps;
    })
    .catch(err => {
      clearInterval(interval);
      alert('Error: ' + err);
    })
    .finally(() => {
      btn.disabled = false;
      btn.textContent = 'Process Video';
    });
}

// ── Dataset Browser ──────────────────────────────
let allImages = [];

function loadDatasetImages() {
  const grid = document.getElementById('image-grid');
  grid.innerHTML = '<p class="placeholder">Loading...</p>';

  fetch('/dataset_images')
    .then(r => r.json())
    .then(data => {
      allImages = data.images;
      renderGrid(allImages);
    })
    .catch(() => {
      grid.innerHTML = '<p class="placeholder">Failed to load images</p>';
    });
}

function renderGrid(images) {
  const grid = document.getElementById('image-grid');
  if (images.length === 0) {
    grid.innerHTML = '<p class="placeholder">No images found</p>';
    return;
  }
  grid.innerHTML = images.map(name => `
    <div class="grid-item" onclick="selectDatasetImage('${name}', this)">
      <p>${name}</p>
    </div>
  `).join('');
}

function filterImages(query) {
  const filtered = allImages.filter(name =>
    name.toLowerCase().includes(query.toLowerCase())
  );
  renderGrid(filtered);
}

function selectDatasetImage(filename, el) {
  document.querySelectorAll('.grid-item').forEach(
    g => g.classList.remove('selected')
  );
  el.classList.add('selected');

  document.getElementById('dataset-result').classList.remove('hidden');
  document.getElementById('ds-original').src = '';
  document.getElementById('ds-detected').src = '';
  document.getElementById('ds-count').textContent = '—';
  document.getElementById('ds-time').textContent  = '—';
  document.getElementById('ds-conf').textContent  = '—';

  document.getElementById('ds-detected').style.display = 'none';

  const spinner = document.createElement('div');
  spinner.className = 'spinner';
  spinner.id = 'ds-spinner';
  document.getElementById('ds-detected').after(spinner);

  fetch('/detect_image', {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ filename })
  })
    .then(r => r.json())
    .then(data => {
      const sp = document.getElementById('ds-spinner');
      if (sp) sp.remove();

      if (data.error) {
        alert(data.error);
        return;
      }

      const orig = document.getElementById('ds-original');
      const det  = document.getElementById('ds-detected');

      orig.src = 'data:image/jpeg;base64,' + data.original;
      det.src  = 'data:image/jpeg;base64,' + data.detected;
      det.style.display = 'block';

      document.getElementById('ds-count').textContent = data.count;
      document.getElementById('ds-time').textContent  = data.inference_ms + 'ms';
      document.getElementById('ds-conf').textContent  = data.avg_conf + '%';
    })
    .catch(err => {
      const sp = document.getElementById('ds-spinner');
      if (sp) sp.remove();
      alert('Error: ' + err);
    });
}