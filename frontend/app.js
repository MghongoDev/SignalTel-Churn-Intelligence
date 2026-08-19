const form = document.querySelector('#prediction-form');
const errorMessage = document.querySelector('#form-error');
const resultEmpty = document.querySelector('#result-empty');
const resultContent = document.querySelector('#result-content');

function formDataToPayload() {
  const data = Object.fromEntries(new FormData(form));
  data.SeniorCitizen = Number(data.SeniorCitizen);
  data.tenure = Number(data.tenure);
  data.MonthlyCharges = Number(data.MonthlyCharges);
  data.TotalCharges = Number(data.TotalCharges);
  return data;
}

function showResult(result) {
  const probability = Math.round(result.churn_probability * 100);
  const highRisk = result.prediction === 1;
  document.querySelector('#result-title').textContent = highRisk ? 'At risk' : 'Likely to stay';
  document.querySelector('#signal-icon').textContent = highRisk ? '!' : '✓';
  document.querySelector('#score').textContent = `${probability}%`;
  document.querySelector('#score-note').textContent = `Prediction: ${result.churn_label}`;
  document.querySelector('#meter-fill').style.width = `${probability}%`;
  document.querySelector('#recommendation').textContent = highRisk
    ? 'Prioritize a retention touchpoint. This profile shows a meaningful likelihood of leaving.'
    : 'This customer currently shows a lower likelihood of churn. Keep the relationship moving forward.';
  resultEmpty.hidden = true;
  resultContent.hidden = false;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  errorMessage.textContent = '';
  const button = form.querySelector('button');
  button.disabled = true;
  button.querySelector('span').textContent = 'Assessing profile...';
  try {
    const response = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(formDataToPayload()) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The assessment could not be completed.');
    showResult(data);
  } catch (error) {
    errorMessage.textContent = error.message;
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Run churn assessment';
  }
});

async function loadModelStatus() {
  try {
    const [healthResponse, infoResponse] = await Promise.all([fetch('/health'), fetch('/model-info')]);
    if (!healthResponse.ok || !infoResponse.ok) throw new Error('Unavailable');
    const info = await infoResponse.json();
    document.querySelector('#status-dot').classList.add('ready');
    document.querySelector('#status-text').textContent = 'Model ready';
    document.querySelector('#model-version').textContent = `v${info.version}`;
    const metricKeys = [['F1', 'F1 score'], ['Recall', 'Recall'], ['ROC-AUC', 'ROC-AUC']];
    document.querySelector('#metrics').innerHTML = metricKeys.map(([key, label]) => `<div><strong>${(info.metrics[key] * 100).toFixed(1)}%</strong><span>${label}</span></div>`).join('');
  } catch {
    document.querySelector('#status-text').textContent = 'Model unavailable';
    errorMessage.textContent = 'The model is offline. Start the API to run an assessment.';
  }
}

loadModelStatus();