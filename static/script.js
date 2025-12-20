// Shared UI state
let currentStartDate = null;
let currentEndDate = null;
let timeseriesChart = null;
const THEME_KEY = 'dashboard-theme';

// Table pagination state
let tableData = [];
let currentPage = 1;
const rowsPerPage = 10;
let tableTickers = [];

function getSelectedTickersFromUI() {
    const cryptoCheckboxes = document.querySelectorAll('input[name="crypto"]:checked');
    const selectedCryptos = Array.from(cryptoCheckboxes).map(cb => cb.value);
    return selectedCryptos.map(sym => String(sym).split('-')[0].toUpperCase());
}

// Boot the dashboard once the DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    initializeDatePicker();
    loadSummary();
    loadTimeSeriesChart();
    loadChartImage('/api/chart/sentiment-distribution', 'sentimentDistImg');
    loadChartImage('/api/chart/sentiment-regimes', 'sentimentRegimesImg');
});

function initTheme() {
    const saved = localStorage.getItem(THEME_KEY) || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    const toggle = document.getElementById('themeSwitch');
    if (toggle) toggle.checked = saved === 'light';
}

function toggleTheme(event) {
    const next = event.target.checked ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem(THEME_KEY, next);
}

// Configure date inputs using the server-provided range/current selection
async function initializeDatePicker() {
    try {
        const response = await fetch('/api/date-range');
        const data = await response.json();
        
        const startInput = document.getElementById('startDate');
        const endInput = document.getElementById('endDate');
        
        currentStartDate = data.current_start;
        currentEndDate = data.current_end;
        
        // Keep the picker flexible; the backend will validate at request time.
        startInput.min = data.min_date;  // 2015-01-01 or earlier
        startInput.max = data.max_date;  // Today's date
        startInput.value = data.current_start;
        
        endInput.min = data.min_date;  // 2015-01-01 or earlier
        endInput.max = data.max_date;  // Today's date
        endInput.value = data.current_end;
    } catch (error) {
        console.error('Error initializing date picker:', error);
    }
}

// Run the notebook pipeline, then refresh all charts/cards
async function fetchDataAndUpdate() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (!startDate || !endDate) {
        showError('Please select both start and end dates');
        return;
    }
    
    if (new Date(startDate) >= new Date(endDate)) {
        showError('Start date must be before end date');
        return;
    }
    
    // Get selected cryptocurrencies
    const cryptoCheckboxes = document.querySelectorAll('input[name="crypto"]:checked');
    const selectedCryptos = Array.from(cryptoCheckboxes).map(cb => cb.value);
    
    if (selectedCryptos.length === 0) {
        showError('Please select at least one cryptocurrency');
        return;
    }
    
    showLoading(true);
    hideError();
    
    try {
        // 1) Run the notebook pipeline (01 -> 02) server-side so the processed CSV matches
        //    the selected assets and date window.
        const pipelineResp = await fetch('/api/run-pipeline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                start_date: startDate,
                end_date: endDate,
                crypto_symbols: selectedCryptos
            })
        });

        const pipelineResult = await pipelineResp.json();
        if (!pipelineResult.success) {
            showError(pipelineResult.error || 'Error running pipeline');
            showLoading(false);
            return;
        }

        // 2) Apply the selection in the app (validation + filtering)
        const response = await fetch('/api/fetch-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                start_date: startDate,
                end_date: endDate,
                crypto_symbols: selectedCryptos
            })
        });

        const result = await response.json();

        if (!result.success) {
            showError(result.error || 'Error fetching data');
            showLoading(false);
            return;
        }
        
        // Persist selection for subsequent requests
        currentStartDate = startDate;
        currentEndDate = endDate;
        
        // Refresh the UI
        await updateAllVisualizations(startDate, endDate);
        
        showLoading(false);
    } catch (error) {
        console.error('Error fetching data:', error);
        showError('Error running pipeline. Please try again.');
        showLoading(false);
    }
}

// Reset back to the current defaults
async function resetToDefaults() {
    const response = await fetch('/api/date-range');
    const data = await response.json();
    
    document.getElementById('startDate').value = data.current_start;
    document.getElementById('endDate').value = data.current_end;
    
    currentStartDate = data.current_start;
    currentEndDate = data.current_end;
    
    // Reload using defaults
    await updateAllVisualizations(null, null);
}

// Fan out the requests needed to redraw the page
async function updateAllVisualizations(startDate, endDate) {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    
    const paramStr = params.toString();
    const query = paramStr ? '?' + paramStr : '';
    
    await Promise.all([
        loadSummary(startDate, endDate),
        loadTimeSeriesChart(startDate, endDate),
        loadChartImage(`/api/chart/sentiment-distribution${query}`, 'sentimentDistImg'),
        loadChartImage(`/api/chart/sentiment-regimes${query}`, 'sentimentRegimesImg')
    ]);
    
    // Only load heavier charts when the tab is visible
    if (document.getElementById('analysis').classList.contains('active')) {
        loadAnalysisCharts(startDate, endDate);
    }
}

// UI helpers
function showLoading(show) {
    const msg = document.getElementById('loadingMessage');
    if (show) {
        msg.style.display = 'flex';
    } else {
        msg.style.display = 'none';
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

// Summary cards
async function loadSummary(startDate = null, endDate = null) {
    try {
        let url = '/api/summary';
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        const paramStr = params.toString();
        if (paramStr) url += '?' + paramStr;
        
        const response = await fetch(url);
        const data = await response.json();
        
        const assets = data.assets || {};
        const tickers = (data.tickers || Object.keys(assets) || []).filter(t => assets[t]);

        const coverageCard = `
            <div class="summary-card">
                <h3>Data Coverage</h3>
                <div class="value">${data.total_days ?? ''}</div>
                <div class="subtext">days analyzed</div>
                <div class="subtext">${data.date_range ?? ''}</div>
            </div>
        `;

        const sentimentCard = `
            <div class="summary-card">
                <h3>Market Sentiment</h3>
                <div class="value">${data.sentiment_current ?? ''}</div>
                <div class="subtext">Current FG Index</div>
                <div class="subtext">Average: ${data.sentiment_avg ?? ''}</div>
            </div>
        `;

        const assetCards = tickers.map(t => {
            const a = assets[t] || {};
            return `
                <div class="summary-card">
                    <h3>${t}</h3>
                    <div class="value">${a.current ?? ''}</div>
                    <div class="subtext">High: ${a.high ?? ''}</div>
                    <div class="subtext">Low: ${a.low ?? ''}</div>
                    <div class="subtext">Avg return: ${a.avg_return ?? ''}</div>
                </div>
            `;
        }).join('');

        const summaryHTML = coverageCard + assetCards + sentimentCard;
        
        document.getElementById('summaryCards').innerHTML = summaryHTML;
    } catch (error) {
        console.error('Error loading summary:', error);
        document.getElementById('summaryCards').innerHTML = '<div class="loading">Error loading summary data</div>';
    }
}

// Main time series (prices + Fear & Greed)
async function loadTimeSeriesChart(startDate = null, endDate = null) {
    try {
        let url = '/api/timeseries';
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        const paramStr = params.toString();
        if (paramStr) url += '?' + paramStr;
        
        const response = await fetch(url);
        const data = await response.json();
        
        const tickers = data.tickers || Object.keys(data.prices || {});
        const prices = data.prices || {};

        // Downsample a bit to keep the plot readable on long ranges
        const step = Math.ceil(data.dates.length / 100);
        const dates = data.dates.filter((_, i) => i % step === 0);
        const sentiment = (data.sentiment || []).filter((_, i) => i % step === 0);
        
        const ctx = document.getElementById('timeseriesChart').getContext('2d');
        
        // Recreate the chart cleanly on every refresh
        if (timeseriesChart) {
            timeseriesChart.destroy();
        }
        
        const palette = ['#00b7ff', '#00c48c', '#ff7f0e', '#f45b69', '#8ea2c2', '#7b61ff', '#00c48c', '#00b7ff'];

        // Subtle fill for the first asset line (keeps the chart from looking too flat)
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(102, 126, 234, 0.3)');
        gradient.addColorStop(1, 'rgba(102, 126, 234, 0)');

        const priceDatasets = tickers.map((t, i) => {
            const series = (prices[t] || []).filter((_, idx) => idx % step === 0);
            return {
                label: `${t} Price (USD)`,
                data: series,
                borderColor: palette[i % palette.length],
                backgroundColor: i === 0 ? gradient : undefined,
                borderWidth: 2,
                fill: i === 0,
                tension: 0.4,
                pointRadius: 0,
                yAxisID: 'y'
            };
        });

        // If one asset dwarfs the others, smaller lines can look flat on a linear axis.
        // Switch to log scale automatically when the spread is large.
        const allPriceValues = [];
        tickers.forEach(t => {
            const series = (prices[t] || []).filter((_, idx) => idx % step === 0);
            for (const v of series) {
                const num = Number(v);
                if (Number.isFinite(num) && num > 0) allPriceValues.push(num);
            }
        });
        const minPrice = allPriceValues.length ? Math.min(...allPriceValues) : null;
        const maxPrice = allPriceValues.length ? Math.max(...allPriceValues) : null;
        const useLogScale = (
            tickers.length >= 2 &&
            minPrice !== null &&
            maxPrice !== null &&
            minPrice > 0 &&
            (maxPrice / minPrice) >= 20
        );
        
        timeseriesChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    ...priceDatasets,
                    {
                        label: 'Fear & Greed Index',
                        data: sentiment,
                        borderColor: '#ff7f0e',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    }
                },
                scales: {
                    y: {
                        type: useLogScale ? 'logarithmic' : 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: useLogScale ? 'Price (USD, log scale)' : 'Price (USD)'
                        },
                        ticks: {
                            callback: function(value) {
                                // Keep ticks readable when using log scale
                                const v = Number(value);
                                if (!Number.isFinite(v)) return value;
                                if (!useLogScale) return v.toLocaleString();
                                if (v >= 1_000_000) return (v / 1_000_000).toFixed(0) + 'M';
                                if (v >= 1_000) return (v / 1_000).toFixed(0) + 'K';
                                return String(v);
                            }
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Sentiment Index'
                        },
                        min: 0,
                        max: 100,
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading time series:', error);
    }
}

function renderPriceDistributions(tickers, startDate = null, endDate = null) {
    const container = document.getElementById('priceDistributions');
    if (!container) return;

    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    const query = params.toString() ? '&' + params.toString() : '';

    container.innerHTML = tickers.map(t => {
        const imgId = `dist_${t}`;
        return `
            <section class="chart-section">
                <h2>${t} Price Distribution</h2>
                <div class="chart-image">
                    <img id="${imgId}" src="" alt="${t} Distribution">
                </div>
            </section>
        `;
    }).join('');

    tickers.forEach(t => {
        loadChartImage(`/api/chart/price-distribution?ticker=${encodeURIComponent(t)}${query}`, `dist_${t}`);
    });
}

// PNG chart endpoints (base64)
async function loadChartImage(apiEndpoint, elementId) {
    try {
        const response = await fetch(apiEndpoint);
        const data = await response.json().catch(() => ({}));

        if (!response.ok || !data || !data.image) {
            const msg = (data && data.error) ? data.error : `Chart not available (${response.status})`;
            throw new Error(msg);
        }

        const img = document.getElementById(elementId);
        img.src = 'data:image/png;base64,' + data.image;
    } catch (error) {
        console.error(`Error loading chart ${elementId}:`, error);
        const img = document.getElementById(elementId);
        if (img) {
            img.removeAttribute('src');
            img.alt = (error && error.message) ? error.message : 'Error loading chart';
        }
    }
}

// Data table
async function loadDataTable(startDate = null, endDate = null) {
    try {
        let url = '/api/table/latest';
        const params = new URLSearchParams();
        
        // Table-specific filters (override the global date range)
        const tableStartDate = document.getElementById('tableStartDate')?.value;
        const tableEndDate = document.getElementById('tableEndDate')?.value;
        const rowCount = document.getElementById('rowCount')?.value || '10';
        
        // Use table dates if set, otherwise fall back to the current dashboard range
        if (tableStartDate) {
            params.append('start_date', tableStartDate);
        } else if (startDate) {
            params.append('start_date', startDate);
        }
        
        if (tableEndDate) {
            params.append('end_date', tableEndDate);
        } else if (endDate) {
            params.append('end_date', endDate);
        }
        
        params.append('limit', rowCount);
        
        const paramStr = params.toString();
        if (paramStr) url += '?' + paramStr;
        
        const response = await fetch(url);
        const payload = await response.json();
        const tickers = payload.tickers || [];
        const rows = payload.rows || [];

        if (rows.length === 0) {
            document.getElementById('tableBody').innerHTML = 
                '<tr><td colspan="7" class="loading">No data available</td></tr>';
            document.getElementById('paginationControls').style.display = 'none';
            return;
        }

        tableTickers = tickers;
        // Cache the payload and reset pagination
        tableData = rows;
        currentPage = 1;

        // Render header
        renderTableHeader();
        
        // Render first page
        renderTablePage();
        
    } catch (error) {
        console.error('Error loading data table:', error);
        document.getElementById('tableBody').innerHTML = 
            '<tr><td colspan="7" class="loading">Error loading data</td></tr>';
        document.getElementById('paginationControls').style.display = 'none';
    }
}

function renderTableHeader() {
    const thead = document.getElementById('tableHead');
    if (!thead) return;

    const tickers = tableTickers || [];

    const cols = [
        '<th>Date</th>',
        ...tickers.map(t => `<th>${t} Price</th>`),
        ...tickers.map(t => `<th>${t} Return</th>`),
        '<th>Sentiment</th>',
        '<th>Classification</th>'
    ];

    thead.innerHTML = `<tr>${cols.join('')}</tr>`;
}

// Render current page of table
function renderTablePage() {
    const tbody = document.getElementById('tableBody');
    const totalPages = Math.ceil(tableData.length / rowsPerPage);
    
    // Calculate start and end indices
    const startIdx = (currentPage - 1) * rowsPerPage;
    const endIdx = Math.min(startIdx + rowsPerPage, tableData.length);
    
    // Get current page data
    const pageData = tableData.slice(startIdx, endIdx);
    
    // Render rows
    const tickers = tableTickers || [];
    tbody.innerHTML = pageData.map(row => {
        const priceCells = tickers.map(t => `<td>${(row.prices && row.prices[t]) ? row.prices[t] : ''}</td>`).join('');
        const returnCells = tickers.map(t => `<td>${(row.returns && row.returns[t]) ? row.returns[t] : ''}</td>`).join('');
        const cls = row.classification || '';
        const clsClass = cls ? `classification-${String(cls).toLowerCase().replace(/\s+/g, '-')}` : '';
        return `
            <tr>
                <td>${row.date ?? ''}</td>
                ${priceCells}
                ${returnCells}
                <td>${row.sentiment ?? ''}</td>
                <td>
                    <span class="${clsClass}">${cls}</span>
                </td>
            </tr>
        `;
    }).join('');
    
    // Update pagination controls
    updatePaginationControls(totalPages);
}

// Update pagination controls
function updatePaginationControls(totalPages) {
    const paginationControls = document.getElementById('paginationControls');
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    const pageInfo = document.getElementById('pageInfo');
    
    if (totalPages <= 1) {
        paginationControls.style.display = 'none';
        return;
    }
    
    paginationControls.style.display = 'flex';
    pageInfo.textContent = `Page ${currentPage} of ${totalPages} (${tableData.length} rows)`;
    
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === totalPages;
}

// Navigate to previous page
function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        renderTablePage();
    }
}

// Navigate to next page
function nextPage() {
    const totalPages = Math.ceil(tableData.length / rowsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        renderTablePage();
    }
}

// Update table data based on filters
function updateTableData() {
    loadDataTable(currentStartDate, currentEndDate);
}

// Reset table filters
function resetTableFilters() {
    document.getElementById('rowCount').value = '10';
    document.getElementById('tableStartDate').value = '';
    document.getElementById('tableEndDate').value = '';
    loadDataTable(currentStartDate, currentEndDate);
}

// Tab switching
function showTab(tabName) {
    // Hide all tab contents
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active'));
    
    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(button => button.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Load charts when switching to analysis tab
    if (tabName === 'analysis') {
        loadAnalysisCharts(currentStartDate, currentEndDate);
    }
    
    // Load table when switching to data tab
    if (tabName === 'data') {
        loadDataTable(currentStartDate, currentEndDate);
    }
}

// Load all analysis charts
function loadAnalysisCharts(startDate = null, endDate = null) {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    const query = params.toString() ? '?' + params.toString() : '';
    
    const tickers = getSelectedTickersFromUI();
    renderPriceDistributions(tickers, startDate, endDate);
    loadChartImage(`/api/chart/returns-vs-sentiment${query}`, 'returnsVsSentimentImg');
    loadChartImage(`/api/chart/volatility${query}`, 'volatilityImg');
    loadChartImage(`/api/chart/correlation${query}`, 'correlationImg');
}
