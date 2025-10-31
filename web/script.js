const audioInput = document.getElementById('audio');
const player = document.getElementById('player');
const fileInfo = document.getElementById('fileInfo');
const transcriptEl = document.getElementById('transcript');
const weightEl = document.getElementById('audio_weight');
const predictBtn = document.getElementById('predictBtn');
const resultCard = document.getElementById('result');
const finalLabel = document.getElementById('finalLabel');
const finalScore = document.getElementById('finalScore');
const fusionList = document.getElementById('fusionList');
const audioList = document.getElementById('audioList');
const textList = document.getElementById('textList');

let audioFile = null;

audioInput.addEventListener('change', () => {
  const file = audioInput.files[0];
  if (!file) {
    audioFile = null;
    player.style.display = 'none';
    fileInfo.textContent = '';
    return;
  }
  audioFile = file;
  const url = URL.createObjectURL(file);
  player.src = url;
  player.style.display = 'block';
  fileInfo.textContent = `✓ Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
});

predictBtn.addEventListener('click', async () => {
  if (!audioFile) {
    alert('⚠️ Please select an audio file first.');
    return;
  }
  predictBtn.disabled = true;
  predictBtn.textContent = '⏳ Analyzing...';
  resultCard.style.display = 'none';

  try {
    const fd = new FormData();
    fd.append('file', audioFile);
    if (transcriptEl.value.trim()) fd.append('transcript', transcriptEl.value.trim());
    fd.append('audio_weight', weightEl.value || '0.7');

    const resp = await fetch('http://127.0.0.1:8000/predict', { method: 'POST', body: fd });
    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`Server error: ${resp.status} - ${errText}`);
    }
    const data = await resp.json();

    finalLabel.textContent = data.final_label;
    finalScore.textContent = data.final_score.toFixed(3);

    const fillList = (el, arr) => {
      el.innerHTML = '';
      if (!arr || !arr.length) { el.innerHTML = '<li>(None)</li>'; return; }
      arr.forEach(item => {
        const li = document.createElement('li');
        li.textContent = `${item.label}: ${(item.score*100).toFixed(1)}%`;
        el.appendChild(li);
      });
    };

    fillList(fusionList, data.top_k);
    fillList(audioList, data.audio_top_k);
    fillList(textList, data.text_top_k);

    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    console.error(e);
    alert(`❌ Prediction error:\n${e.message}\n\nPlease check:\n1. Backend is running (http://127.0.0.1:8000)\n2. Audio file is valid (.wav/.mp3)`);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict Emotion';
  }
});
