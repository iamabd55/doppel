// ForgeryDetect client application
document.addEventListener('DOMContentLoaded', () => {
  // ════════════ Lenis Smooth Scroll ════════════
  let lenis = null;
  if (typeof Lenis !== 'undefined') {
    lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: 'vertical',
      gestureOrientation: 'vertical',
      smoothWheel: true,
      smoothTouch: false,
      wheelMultiplier: 1.0,
    });

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    // Smooth scroll for anchor navigation links (#main, #features)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        const targetId = this.getAttribute('href');
        if (targetId && targetId !== '#') {
          const target = document.querySelector(targetId);
          if (target) {
            e.preventDefault();
            lenis.scrollTo(target, { offset: -60, duration: 1.2 });
          }
        }
      });
    });
  }

  // Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const dropContent = document.getElementById('dropContent');
  const dropPreview = document.getElementById('dropPreview');
  const previewImg = document.getElementById('previewImg');
  const clearBtn = document.getElementById('clearBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');

  const comofodCategory = document.getElementById('comofodCategory');
  const comofodSelect = document.getElementById('comofodSelect');
  const loadComofodBtn = document.getElementById('loadComofod');

  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsContent = document.getElementById('resultsContent');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const statusBadge = document.getElementById('statusBadge');
  const resultImage = document.getElementById('resultImage');

  const statFlagged = document.getElementById('statFlagged');
  const statPct = document.getElementById('statPct');
  const statTime = document.getElementById('statTime');
  const statRegions = document.getElementById('statRegions');

  const shiftsSection = document.getElementById('shiftsSection');
  const shiftsBody = document.getElementById('shiftsBody');
  const downloadMaskBtn = document.getElementById('downloadMask');

  const tabs = document.querySelectorAll('.tab');

  // State
  let currentFile = null;
  let currentResults = null;
  let categoryMap = {};

  // ════════════ Drag & Drop ════════════
  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleFileSelected(files[0]);
    }
  });

  dropZone.addEventListener('click', (e) => {
    if (e.target !== clearBtn && !currentFile) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImage();
  });

  function handleFileSelected(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropContent.style.display = 'none';
      dropPreview.style.display = 'flex';
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function clearImage() {
    currentFile = null;
    fileInput.value = '';
    previewImg.src = '';
    dropContent.style.display = 'flex';
    dropPreview.style.display = 'none';
    analyzeBtn.disabled = true;
  }

  // ════════════ CoMoFoD List & Load ════════════
  async function loadComofodList() {
    try {
      const res = await fetch('/api/comofod/list');
      const data = await res.json();
      if (data.categories) {
        categoryMap = data.categories;
        updateImageListForCategory(comofodCategory ? comofodCategory.value : 'translation');
      }
    } catch (err) {
      console.warn('Could not load CoMoFoD list:', err);
    }
  }

  function updateImageListForCategory(cat) {
    comofodSelect.innerHTML = '';
    const list = categoryMap[cat] || [];
    if (list.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '— No images —';
      comofodSelect.appendChild(opt);
      return;
    }
    list.forEach(num => {
      const opt = document.createElement('option');
      const tag = String(num).padStart(3, '0');
      opt.value = num;
      opt.textContent = `${tag}_F.png`;
      comofodSelect.appendChild(opt);
    });
  }

  if (comofodCategory) {
    comofodCategory.addEventListener('change', () => {
      updateImageListForCategory(comofodCategory.value);
    });
  }
  loadComofodList();

  loadComofodBtn.addEventListener('click', async () => {
    const val = comofodSelect.value;
    if (!val) return;

    loadComofodBtn.disabled = true;
    loadComofodBtn.textContent = 'Loading...';
    try {
      const res = await fetch(`/api/comofod/${val}`);
      const data = await res.json();
      if (data.image) {
        // Convert base64 to Blob / File
        const byteCharacters = atob(data.image);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/png' });
        const file = new File([blob], data.filename || 'comofod.png', { type: 'image/png' });

        handleFileSelected(file);
      }
    } catch (err) {
      alert('Error loading sample image: ' + err.message);
    } finally {
      loadComofodBtn.disabled = false;
      loadComofodBtn.textContent = 'Load';
    }
  });

  // ════════════ Analysis ════════════
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Show loading
    loadingOverlay.style.display = 'flex';
    analyzeBtn.disabled = true;
    statusBadge.textContent = '● Analyzing...';
    statusBadge.className = 'badge badge-red';

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('Q', document.getElementById('paramQ').value);
    formData.append('min_std', document.getElementById('paramMinStd').value);
    formData.append('K', document.getElementById('paramK').value);
    formData.append('min_votes', document.getElementById('paramMinVotes').value);
    formData.append('min_dist', document.getElementById('paramMinDist').value);
    formData.append('window', document.getElementById('paramWindow').value);
    formData.append('kernel', document.getElementById('paramKernel').value);
    formData.append('min_size', document.getElementById('paramMinSize').value);

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Server error occurred during detection');
      }

      currentResults = data;
      renderResults(data);
      statusBadge.textContent = '● Complete';
      statusBadge.className = 'badge badge-green';
    } catch (err) {
      alert('Detection failed: ' + err.message);
      statusBadge.textContent = '● Error';
      statusBadge.className = 'badge badge-red';
    } finally {
      loadingOverlay.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  });

  function renderResults(data) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'flex';

    // Set stats
    statFlagged.textContent = data.stats.flagged.toLocaleString();
    statPct.textContent = data.stats.pct + '%';
    statTime.textContent = data.stats.time + 's';
    statRegions.textContent = data.stats.regions;

    // Set image to default (Overlay)
    setActiveTab('overlay');

    // Populate shift vectors
    if (data.shifts && data.shifts.length > 0) {
      shiftsBody.innerHTML = '';
      data.shifts.forEach(s => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${s.dy > 0 ? '+' : ''}${s.dy}</td><td>${s.dx > 0 ? '+' : ''}${s.dx}</td><td><strong>${s.votes}</strong></td>`;
        shiftsBody.appendChild(tr);
      });
      shiftsSection.style.display = 'block';
    } else {
      shiftsSection.style.display = 'none';
    }
  }

  // ════════════ View Tabs (Overlay / Mask / Original) ════════════
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const view = tab.getAttribute('data-view');
      setActiveTab(view);
    });
  });

  function setActiveTab(view) {
    tabs.forEach(t => t.classList.toggle('active', t.getAttribute('data-view') === view));
    if (currentResults && currentResults[view]) {
      resultImage.src = 'data:image/png;base64,' + currentResults[view];
    }
  }

  // ════════════ Download Mask ════════════
  downloadMaskBtn.addEventListener('click', () => {
    if (!currentResults || !currentResults.mask) return;
    const a = document.createElement('a');
    a.href = 'data:image/png;base64,' + currentResults.mask;
    a.download = 'forgery_mask.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });
});

