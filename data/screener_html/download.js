const fs = require("fs");
const path = require("path");

const tickers = `
BIRET EMBASSY SWARAJENG TATACHEM TEAMLEASE RAMCOCEM THERMAX TIMKEN ASIANPAINT
`.trim().split(/\s+/);

async function downloadTicker(ticker) {
    const url = `https://www.screener.in/company/${ticker}/`;

    try {
        const response = await fetch(url, {
            headers: {
                "User-Agent": "Mozilla/5.0"
            }
        });

        const html = await response.text();

        fs.writeFileSync(
            path.join(__dirname, `${ticker}.html`),
            html
        );

        console.log(`Saved ${ticker}.html`);
    } catch (err) {
        console.error(`Failed ${ticker}`, err.message);
    }
}

(async () => {
    for (const ticker of tickers) {
        await downloadTicker(ticker);

        // avoid hammering server
        await new Promise(r => setTimeout(r, 15000));
    }
})();
