document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const loading = document.getElementById('loading');
    const errorBox = document.getElementById('error-box');
    const resultBox = document.getElementById('result-box');
    const sampleBtns = document.querySelectorAll('.sample-btn');

    const resultEmoji = document.getElementById('result-emoji');
    const resultEmotion = document.getElementById('result-emotion');
    const resultConfidence = document.getElementById('result-confidence');
    const nlpTokens = document.getElementById('nlp-tokens');
    const breakdownList = document.getElementById('breakdown-list');

    // Load Metrics on start
    loadMetrics();

    // Sample Button Clicks
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            textInput.value = btn.getAttribute('data-text');
            analyzeEmotion();
        });
    });

    // Clear Button Click
    clearBtn.addEventListener('click', () => {
        textInput.value = '';
        resultBox.classList.add('hidden');
        errorBox.classList.add('hidden');
        textInput.focus();
    });

    // Analyze Button Click
    analyzeBtn.addEventListener('click', analyzeEmotion);

    // Enter Key shortcut (Ctrl+Enter or Cmd+Enter)
    textInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            analyzeEmotion();
        }
    });

    async function analyzeEmotion() {
        const text = textInput.value.trim();
        if (!text) {
            showError('Please enter some text to analyze.');
            return;
        }

        hideError();
        resultBox.classList.add('hidden');
        loading.classList.remove('hidden');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze text.');
            }

            displayResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            loading.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    }

    function displayResult(data) {
        resultEmoji.textContent = data.emoji;
        resultEmotion.textContent = data.emotion;
        resultEmotion.style.color = data.color;
        resultConfidence.textContent = `${data.confidence_percentage}%`;
        nlpTokens.textContent = data.cleaned_text || '(None)';

        // Render Breakdown Progress Bars
        breakdownList.innerHTML = '';
        data.breakdown.forEach(item => {
            const row = document.createElement('div');
            row.className = 'breakdown-item';
            row.innerHTML = `
                <div class="breakdown-label">
                    <span>${item.emoji} ${item.emotion}</span>
                    <span>${item.percentage}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${item.percentage}%; background-color: ${item.color};"></div>
                </div>
            `;
            breakdownList.appendChild(row);
        });

        resultBox.classList.remove('hidden');
    }

    async function loadMetrics() {
        try {
            const res = await fetch('/api/metrics');
            if (!res.ok) return;
            const metrics = await res.json();

            document.getElementById('metric-model').textContent = metrics.best_model || 'Linear SVM';
            document.getElementById('metric-acc').textContent = metrics.test_accuracy ? `${(metrics.test_accuracy * 100).toFixed(2)}%` : '90.35%';

            const tbody = document.getElementById('comparison-table-body');
            tbody.innerHTML = '';

            for (const [modelName, info] of Object.entries(metrics.comparison || {})) {
                const tr = document.createElement('tr');
                if (modelName === metrics.best_model) {
                    tr.className = 'best-row';
                }
                tr.innerHTML = `
                    <td>${modelName} ${modelName === metrics.best_model ? '🏆' : ''}</td>
                    <td>${(info.val_accuracy * 100).toFixed(2)}%</td>
                    <td>${(info.test_accuracy * 100).toFixed(2)}%</td>
                    <td>${(info.f1_score * 100).toFixed(2)}%</td>
                `;
                tbody.appendChild(tr);
            }
        } catch (e) {
            console.log('Metrics loading failed:', e);
        }
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.remove('hidden');
    }

    function hideError() {
        errorBox.textContent = '';
        errorBox.classList.add('hidden');
    }
});
